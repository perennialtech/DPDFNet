#define _GNU_SOURCE
#include "worker.h"
#include <immintrin.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <x86intrin.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

/* Spin budget in TSC ticks (about 1 ms at typical clocks). Within a hop,
 * waits are shorter: sleeping there would add a kernel wake-up, which costs
 * tens of microseconds once a core idles. Past the budget, waiting threads
 * sleep so an oversubscribed core is not monopolized by spinning. */
enum { SPIN=1u<<22 };
/* Spinning threads yield every YIELD ticks (about 20 us). A wake-up can place
 * the helper on the caller's CPU and preempt it; without yielding, whichever
 * thread spins there would hold the CPU its partner needs for the whole
 * budget. Without a waiting thread, the yield returns at once. */
enum { YIELD=1u<<16 };
enum { IDLE, POSTED, RUNNING, DONE };
/* Pre-wake: with a steady hop cadence the helper sets a timer to wake ahead
 * of the next expected hop and spins until it starts, or until WINDOW plus
 * twice the hop jitter (at most MAX_LEAD) after it. The lead covers the
 * timer's observed lateness, twice the hop jitter and MARGIN, within
 * [MIN_LEAD,MAX_LEAD], so one pre-wake spins for at most 800 us (1.3 ms on
 * Windows, see TICK). Cadences outside [MIN_PERIOD,MAX_PERIOD] or with jitter
 * above an eighth of the period are not predicted. */
static const int64_t MARGIN=30000,MIN_LEAD=40000,MAX_LEAD=300000,WINDOW=200000,
                     MIN_PERIOD=100000,MAX_PERIOD=100000000;
/* Windows expires timers, high-resolution ones included, on clock interrupts,
 * which come at most every 0.5 ms, so a timer fires up to TICK late. With a
 * hop cadence near a multiple of the tick, that lateness repeats hop after
 * hop, and where it exceeds MAX_LEAD no pre-wake ever lands. The helper
 * therefore sets its timer one tick early, so that a tick falls before the
 * lead, and spins the rest. */
#ifdef _WIN32
static const int64_t TICK=500000;
#else
static const int64_t TICK=0;
#endif
/* Back-off. A hop loses when the caller waited for a task longer than the
 * task usually takes to run (the helper was preempted holding it): that adds
 * SLOW to a score. A hop where the helper started no task, so the caller ran
 * them all, gained nothing: that adds IDLE_HOP. Any other hop takes one off. Past
 * LIMIT (sustained slow hops above about one in nine, or idle ones above one
 * in three) the next hops run on the caller alone. The pause starts at
 * MIN_PAUSE hops and doubles, up to MAX_PAUSE, while losses resume within
 * RECENT hops. */
enum { SLOW=8, IDLE_HOP=2, LIMIT=32, MIN_PAUSE=64, MAX_PAUSE=4096, RECENT=1024 };

struct dpdf_worker {
    _Alignas(64) atomic_uint task; /* Task phase; the caller sleeps on it. */
    atomic_uint caller_waiting;
    _Alignas(64) atomic_uint wake; /* Bumped to wake the helper. */
    atomic_uint active,sleeping,quit;
    _Atomic int64_t next,jitter2; /* Predicted next hop (ns), 0 if none. */
    dpdf_task fn;
    void *arg;
    unsigned csr,flags;
    int status;
    int64_t last,period,jitter; /* Caller-side cadence estimate. */
    int outliers;
    uint64_t ran,usual[4]; /* Helper task run time; typical run time of each
                            * of a hop's tasks, in posting order (TSC). */
    int loss,joins,stolen,hopped,score,paused,pause,since; /* Caller-side back-off. */
#ifdef _WIN32
    HANDLE thread;
    /* Auto-reset events stand in for futexes on task and wake: a signal
     * stays set until a wait consumes it, so one sent before the wait is not
     * lost, and every waiter rechecks its condition after waking. */
    HANDLE task_event,wake_event;
    HANDLE timer; /* High-resolution: the default timer tick is 15.6 ms. */
#else
    pthread_t thread;
    pid_t pid; /* The creating process; a forked child has no helper. */
#endif
};

#ifdef _WIN32
static int64_t now_ns(void) {
    LARGE_INTEGER f,c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return c.QuadPart/f.QuadPart*1000000000+c.QuadPart%f.QuadPart*1000000000/f.QuadPart;
}
static HANDLE event_for(dpdf_worker *w,atomic_uint *word) {
    return word==&w->task ? w->task_event : w->wake_event;
}
static void park(dpdf_worker *w,atomic_uint *word,unsigned value,int64_t timeout) {
    if (atomic_load_explicit(word,memory_order_seq_cst)!=value) return;
    if (timeout>0) {
        /* Relative due time in 100 ns units, rounded up. */
        LARGE_INTEGER due={.QuadPart=-((timeout+99)/100)};
        if (SetWaitableTimer(w->timer,&due,0,NULL,NULL,FALSE)) {
            HANDLE handles[2]={event_for(w,word),w->timer};
            WaitForMultipleObjects(2,handles,FALSE,INFINITE);
            return;
        }
    }
    WaitForSingleObject(event_for(w,word),INFINITE);
}
static void unpark(dpdf_worker *w,atomic_uint *word) { SetEvent(event_for(w,word)); }
static void yield(void) { SwitchToThread(); }
/* Windows has no fork. */
static int forked(const dpdf_worker *w) { (void)w; return 0; }
#else
static int64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return (int64_t)t.tv_sec*1000000000+t.tv_nsec;
}
static void park(dpdf_worker *w,atomic_uint *word,unsigned value,int64_t timeout) {
    (void)w;
    struct timespec t={(time_t)(timeout/1000000000),(long)(timeout%1000000000)};
    syscall(SYS_futex,(void *)word,FUTEX_WAIT_PRIVATE,value,timeout>0 ? &t : NULL,NULL,0);
}
static void unpark(dpdf_worker *w,atomic_uint *word) {
    (void)w;
    syscall(SYS_futex,(void *)word,FUTEX_WAKE_PRIVATE,1,NULL,NULL,0);
}
static void yield(void) { sched_yield(); }
static int forked(const dpdf_worker *w) { return getpid()!=w->pid; }
#endif
static void wake_helper(dpdf_worker *w) {
    if (atomic_load_explicit(&w->sleeping,memory_order_seq_cst)) {
        atomic_fetch_add_explicit(&w->wake,1,memory_order_seq_cst);
        unpark(w,&w->wake);
    }
}
/* Sleep until woken (timeout 0) or for timeout ns. Dekker with post and
 * begin: either the recheck sees their store, or they see sleeping and bump
 * wake past seq. With expired, an active hop no longer keeps it awake. */
static void helper_sleep(dpdf_worker *w,int64_t timeout,int expired) {
    const unsigned seq=atomic_load_explicit(&w->wake,memory_order_seq_cst);
    atomic_store_explicit(&w->sleeping,1,memory_order_seq_cst);
    if (atomic_load_explicit(&w->task,memory_order_seq_cst)!=POSTED &&
        !atomic_load_explicit(&w->quit,memory_order_seq_cst) &&
        (expired || !atomic_load_explicit(&w->active,memory_order_seq_cst)))
        park(w,&w->wake,seq,timeout);
    atomic_store_explicit(&w->sleeping,0,memory_order_relaxed);
}
static void relax(uint64_t *mark) {
    _mm_pause();
    const uint64_t now=__rdtsc();
    if (now-*mark>YIELD) { yield(); *mark=now; }
}
static int waiting(dpdf_worker *w) {
    return atomic_load_explicit(&w->task,memory_order_acquire)!=POSTED &&
           !atomic_load_explicit(&w->quit,memory_order_relaxed);
}
/* Returns when a task is posted or quit is set. late/spread: the helper's
 * estimate of how late its timer fires and how much that varies. */
static void helper_idle(dpdf_worker *w,int64_t *late,int64_t *spread) {
    while (waiting(w)) {
        if (atomic_load_explicit(&w->active,memory_order_relaxed)) {
            const uint64_t start=__rdtsc();
            uint64_t mark=start;
            while (waiting(w) && atomic_load_explicit(&w->active,memory_order_relaxed)) {
                if (__rdtsc()-start>SPIN) { helper_sleep(w,0,1); break; }
                relax(&mark);
            }
            continue;
        }
        const int64_t next=atomic_load_explicit(&w->next,memory_order_relaxed),now=now_ns();
        const int64_t jitter2=atomic_load_explicit(&w->jitter2,memory_order_relaxed),close=next+WINDOW+jitter2;
        int64_t lead=*late+2**spread+jitter2+MARGIN;
        lead=lead<MIN_LEAD ? MIN_LEAD : lead>MAX_LEAD ? MAX_LEAD : lead;
        const int64_t due=next-lead-TICK;
        if (next && now<due) {
            helper_sleep(w,due-now,0);
            /* A wake before due was not the timer. One within the tick before
             * the lead is on time; only later ones count as late. */
            const int64_t woke=now_ns(),error=woke-(next-lead);
            if (woke>=due && waiting(w) && !atomic_load_explicit(&w->active,memory_order_relaxed)) {
                const int64_t miss=(error>0 ? error : 0)-*late;
                *late+=miss/8; *spread+=((miss<0 ? -miss : miss)-*spread)/8;
            }
        }
        else if (next && now<close) {
            uint64_t mark=__rdtsc();
            while (waiting(w) && !atomic_load_explicit(&w->active,memory_order_relaxed) && now_ns()<close)
                relax(&mark);
            int64_t seen=next; /* Missed: sleep, unless begin set a new prediction. */
            if (waiting(w) && !atomic_load_explicit(&w->active,memory_order_relaxed))
                atomic_compare_exchange_strong_explicit(&w->next,&seen,0,memory_order_relaxed,memory_order_relaxed);
        } else helper_sleep(w,0,0);
    }
}

static void helper(dpdf_worker *w) {
    int64_t late=50000,spread=0;
#ifndef _WIN32
    prctl(PR_SET_TIMERSLACK,1000UL,0,0,0); /* Pre-wake timers fire on time. */
#endif
    for (;;) {
        helper_idle(w,&late,&spread);
        if (atomic_load_explicit(&w->quit,memory_order_relaxed)) return;
        unsigned expected=POSTED;
        if (!atomic_compare_exchange_strong_explicit(&w->task,&expected,RUNNING,
                memory_order_acquire,memory_order_relaxed)) continue; /* The caller took it. */
        const unsigned own=_mm_getcsr();
        _mm_setcsr(w->csr&~0x3fu);
        const uint64_t start=__rdtsc();
        w->status=w->fn(w->arg);
        w->ran=__rdtsc()-start;
        w->flags=_mm_getcsr()&0x3fu;
        _mm_setcsr(own);
        atomic_store_explicit(&w->task,DONE,memory_order_seq_cst);
        if (atomic_load_explicit(&w->caller_waiting,memory_order_seq_cst)) unpark(w,&w->task);
    }
}

#ifdef _WIN32
static DWORD WINAPI helper_main(LPVOID p) { helper(p); return 0; }
static void release(dpdf_worker *w) {
    if (w->timer) CloseHandle(w->timer);
    if (w->wake_event) CloseHandle(w->wake_event);
    if (w->task_event) CloseHandle(w->task_event);
    _aligned_free(w);
}
/* Kernels keep scratch on the stack (the API asks callers for 1 MiB), so the
 * helper reserves 2 MiB. Windows does not inherit a thread's priority, so the
 * helper copies the creating thread's, as default attributes do on Linux. */
static int start(dpdf_worker *w) {
    w->task_event=CreateEventW(NULL,FALSE,FALSE,NULL);
    w->wake_event=CreateEventW(NULL,FALSE,FALSE,NULL);
    w->timer=CreateWaitableTimerExW(NULL,NULL,CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,TIMER_ALL_ACCESS);
    if (!w->task_event || !w->wake_event || !w->timer) return -1;
    w->thread=CreateThread(NULL,(SIZE_T)2<<20,helper_main,w,
                           STACK_SIZE_PARAM_IS_A_RESERVATION|CREATE_SUSPENDED,NULL);
    if (!w->thread) return -1;
    SetThreadPriority(w->thread,GetThreadPriority(GetCurrentThread()));
    SetThreadDescription(w->thread,L"dpdfnet-lane");
    ResumeThread(w->thread);
    return 0;
}
static void stop(dpdf_worker *w) {
    WaitForSingleObject(w->thread,INFINITE);
    CloseHandle(w->thread);
}
#else
static void *helper_main(void *p) { helper(p); return NULL; }
static void release(dpdf_worker *w) { free(w); }
static int start(dpdf_worker *w) {
    /* The helper inherits a fully blocked signal mask, so process signals
     * keep going to the application's own threads. */
    sigset_t all,old;
    sigfillset(&all);
    /* Kernels keep scratch on the stack (the API asks callers for 1 MiB), so
     * the helper gets at least 2 MiB whatever the process default. Scheduling
     * is inherited from the creating thread, as with default attributes. */
    pthread_attr_t attr;
    size_t stack=0;
    if (pthread_attr_init(&attr)) return -1;
    if (pthread_attr_getstacksize(&attr,&stack) || stack<(size_t)2<<20) pthread_attr_setstacksize(&attr,(size_t)2<<20);
    pthread_sigmask(SIG_SETMASK,&all,&old);
    const int failed=pthread_create(&w->thread,&attr,helper_main,w);
    pthread_sigmask(SIG_SETMASK,&old,NULL);
    pthread_attr_destroy(&attr);
    if (failed) return -1;
    w->pid=getpid();
    pthread_setname_np(w->thread,"dpdfnet-lane");
    return 0;
}
static void stop(dpdf_worker *w) { pthread_join(w->thread,NULL); }
#endif

dpdf_worker *dpdf_worker_create(void) {
    const size_t bytes=(sizeof(dpdf_worker)+63)/64*64;
#ifdef _WIN32
    dpdf_worker *w=_aligned_malloc(bytes,64);
    if (!w) return NULL;
    w->thread=NULL; w->task_event=NULL; w->wake_event=NULL; w->timer=NULL;
#else
    dpdf_worker *w=aligned_alloc(64,bytes);
    if (!w) return NULL;
#endif
    atomic_init(&w->task,IDLE); atomic_init(&w->caller_waiting,0); atomic_init(&w->wake,0);
    atomic_init(&w->active,0); atomic_init(&w->sleeping,0); atomic_init(&w->quit,0);
    atomic_init(&w->next,0); atomic_init(&w->jitter2,0);
    w->fn=NULL; w->arg=NULL; w->csr=0; w->flags=0; w->status=0;
    w->last=0; w->period=0; w->jitter=0; w->outliers=0;
    w->ran=0; for (int i=0;i<4;++i) w->usual[i]=0;
    w->loss=0; w->joins=0; w->stolen=0; w->hopped=0; w->score=0; w->paused=0;
    w->pause=MIN_PAUSE; w->since=0;
    if (start(w)) { release(w); return NULL; }
    return w;
}

void dpdf_worker_destroy(dpdf_worker *w) {
    if (!w) return;
    /* After fork the child has no helper to stop; its stack is leaked. */
    if (forked(w)) { release(w); return; }
    atomic_store_explicit(&w->quit,1,memory_order_seq_cst);
    atomic_fetch_add_explicit(&w->wake,1,memory_order_seq_cst);
    unpark(w,&w->wake);
    stop(w);
    release(w);
}

size_t dpdf_worker_bytes(void) { return (sizeof(dpdf_worker)+63)/64*64; }

int dpdf_worker_begin(dpdf_worker *w) {
    if (forked(w)) return 0; /* A forked child runs hops alone. */
    const int64_t now=now_ns();
    if (w->last) {
        const int64_t interval=now-w->last,error=interval-w->period;
        const int64_t size=error<0 ? -error : error;
        /* An isolated late or early hop (a glitch, a pause) is ignored; three
         * in a row mean the cadence changed. */
        if (!w->period || (size>w->period/2 && ++w->outliers>=3)) {
            w->period=interval; w->jitter=0; w->outliers=0;
        } else if (size<=w->period/2) {
            w->outliers=0; w->period+=error/8; w->jitter+=(size-w->jitter)/8;
        }
    }
    w->last=now;
    const int steady=w->period>=MIN_PERIOD && w->period<=MAX_PERIOD && w->jitter*8<=w->period;
    atomic_store_explicit(&w->jitter2,2*w->jitter<MAX_LEAD ? 2*w->jitter : MAX_LEAD,memory_order_relaxed);
    if (w->hopped) {
        w->hopped=0; ++w->since;
        const int add=w->loss ? SLOW : w->stolen==w->joins ? IDLE_HOP : 0;
        w->score=add ? w->score+add : w->score>0 ? w->score-1 : 0;
        w->loss=0; w->joins=0; w->stolen=0;
        if (w->score>LIMIT) {
            w->pause=w->since<=RECENT ? (w->pause*2<MAX_PAUSE ? w->pause*2 : MAX_PAUSE) : MIN_PAUSE;
            w->paused=w->pause; w->score=0; w->since=0;
        }
    }
    if (w->paused) { /* Pre-wake again only for the first hop after the pause. */
        --w->paused;
        atomic_store_explicit(&w->next,!w->paused && steady ? now+w->period : 0,memory_order_relaxed);
        return 0;
    }
    atomic_store_explicit(&w->next,steady ? now+w->period : 0,memory_order_relaxed);
    atomic_store_explicit(&w->active,1,memory_order_seq_cst);
    wake_helper(w);
    w->hopped=1;
    return 1;
}

void dpdf_worker_pause(dpdf_worker *w,int hops) { w->paused=hops; }

void dpdf_worker_post(dpdf_worker *w,dpdf_task fn,void *arg) {
    w->fn=fn; w->arg=arg; w->csr=_mm_getcsr();
    atomic_store_explicit(&w->task,POSTED,memory_order_seq_cst);
    wake_helper(w);
}

int dpdf_worker_join(dpdf_worker *w) {
    unsigned expected=POSTED;
    ++w->joins;
    if (atomic_compare_exchange_strong_explicit(&w->task,&expected,RUNNING,
            memory_order_acquire,memory_order_relaxed)) {
        const int status=w->fn(w->arg); /* Unclaimed: run it here. */
        atomic_store_explicit(&w->task,IDLE,memory_order_relaxed);
        ++w->stolen;
        return status;
    }
    const uint64_t start=__rdtsc();
    uint64_t mark=start;
    while (atomic_load_explicit(&w->task,memory_order_acquire)!=DONE) {
        if (__rdtsc()-start<SPIN) { relax(&mark); continue; }
        atomic_store_explicit(&w->caller_waiting,1,memory_order_seq_cst);
        if (atomic_load_explicit(&w->task,memory_order_seq_cst)!=DONE) park(w,&w->task,RUNNING,0);
        atomic_store_explicit(&w->caller_waiting,0,memory_order_relaxed);
    }
    const uint64_t waited=__rdtsc()-start;
    /* usual follows the task's run time, ignoring runs over twice it. */
    uint64_t *usual=&w->usual[w->joins<4 ? w->joins-1 : 3];
    if (!*usual) *usual=w->ran;
    else if (w->ran<2**usual) *usual=(uint64_t)((int64_t)*usual+((int64_t)w->ran-(int64_t)*usual)/8);
    if (waited>*usual) w->loss=1;
    _mm_setcsr(_mm_getcsr()|w->flags);
    atomic_store_explicit(&w->task,IDLE,memory_order_relaxed);
    return w->status;
}

void dpdf_worker_end(dpdf_worker *w) {
    atomic_store_explicit(&w->active,0,memory_order_relaxed);
}
