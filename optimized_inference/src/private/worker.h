#ifndef DPDF_WORKER_H
#define DPDF_WORKER_H
#include <stddef.h>
/* One helper thread that runs a single posted task at a time.
 *
 * Within a hop the caller brackets its work with begin/end; end may follow
 * the post of the hop's last task. While a hop is active the helper spins
 * between tasks; otherwise it sleeps in the kernel. When begin arrives at a
 * steady cadence, the helper wakes shortly before the next expected begin
 * and spins until it, so its core is already awake for the first task.
 * A posted task runs on whichever thread claims it first: if the helper has
 * not started it by the time the caller joins, the caller runs it itself, so
 * a descheduled helper never delays a hop beyond single-threaded time.
 *
 * Tasks run with the poster's MXCSR rounding and FTZ/DAZ controls; floating-
 * point exception flags the helper raises are merged into the caller's.
 *
 * begin returns 0 while the worker is backing off, after the helper has kept
 * failing to help (see worker.c): the caller then runs the hop alone and
 * calls neither post nor end. */
typedef struct dpdf_worker dpdf_worker;
typedef int (*dpdf_task)(void *);
dpdf_worker *dpdf_worker_create(void);
void dpdf_worker_destroy(dpdf_worker *);
size_t dpdf_worker_bytes(void);
int dpdf_worker_begin(dpdf_worker *);
/* Test support: run the next hops on the caller alone. */
void dpdf_worker_pause(dpdf_worker *, int hops);
void dpdf_worker_post(dpdf_worker *, dpdf_task, void *);
/* Waits for (or runs) the posted task and returns its result. */
int dpdf_worker_join(dpdf_worker *);
void dpdf_worker_end(dpdf_worker *);
#endif
