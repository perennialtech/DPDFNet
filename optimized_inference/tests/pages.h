#ifndef DPDF_TEST_PAGES_H
#define DPDF_TEST_PAGES_H
/* Page mappings for guard regions: map inaccessible, then open the payload
 * pages read-write or read-only. Protection calls return 0 on success. */
#include <stddef.h>
#ifdef _WIN32
#include <windows.h>
static long page_size(void) { SYSTEM_INFO s; GetSystemInfo(&s); return (long)s.dwPageSize; }
static void *map_none(size_t bytes) {
    return VirtualAlloc(NULL,bytes,MEM_RESERVE|MEM_COMMIT,PAGE_NOACCESS);
}
static int protect_rw(void *p,size_t bytes) { DWORD old; return !VirtualProtect(p,bytes,PAGE_READWRITE,&old); }
static int protect_ro(void *p,size_t bytes) { DWORD old; return !VirtualProtect(p,bytes,PAGE_READONLY,&old); }
static int unmap(void *p,size_t bytes) { (void)bytes; return !VirtualFree(p,0,MEM_RELEASE); }
#else
#include <sys/mman.h>
#include <unistd.h>
static long page_size(void) { return sysconf(_SC_PAGESIZE); }
static void *map_none(size_t bytes) {
    void *p=mmap(NULL,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    return p==MAP_FAILED ? NULL : p;
}
static int protect_rw(void *p,size_t bytes) { return mprotect(p,bytes,PROT_READ|PROT_WRITE); }
static int protect_ro(void *p,size_t bytes) { return mprotect(p,bytes,PROT_READ); }
static int unmap(void *p,size_t bytes) { return munmap(p,bytes); }
#endif
#endif
