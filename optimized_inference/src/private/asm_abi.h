#ifndef DPDF_ASM_ABI_H
#define DPDF_ASM_ABI_H
/* The assembly kernels use the System V x86-64 calling convention on every
 * platform. On Windows, GCC calls them through sysv_abi and saves the
 * registers Win64 treats as callee-saved (RSI, RDI, XMM6-XMM15) once in the
 * calling function's prologue, not around each call. */
#ifdef _WIN32
#define DPDF_SYSV __attribute__((sysv_abi))
#else
#define DPDF_SYSV
#endif
#endif
