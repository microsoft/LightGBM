/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Adapted from: https://www.codeproject.com/Articles/33249/Debugging-C-Code-from-Java-Application
 *
 */
#ifndef LIGHTGBM_UTILS_GDB_H_
#define LIGHTGBM_UTILS_GDB_H_

#ifdef DEBUG
#define GDB() exec_gdb()
#define GDB_ON_TRIGGER() gdb_on_trigger()
#else
#define GDB() {}
#define GDB_BY_TRIGGER() {}
#endif

/*!
 * \brief Allows the next gdb_on_trigger() call to dispatch the debugger.
 */
extern "C" void trigger_gdb();
/*!
 * \brief Call this manually through the debugger after attaching the debugger.
 * Only then will the normal program execution resume.
 */
extern "C" void signal_gdb_attached();

/*!
 * \brief Starts the debugger.
 * The main program will wait for a call from `signal_gdb_attached()` to resume.
 * In case the debugger doesn't successfully launch, periodically a new message tells you
 * the PID of the target PID to attach to, in case you want to attach the debugger
 * manually.
 * Only after the `signal_gdb_attached()` call by the user will the main program resume execution.
 */
void exec_gdb();

/*!
 * \brief Same as `exec_gdb()` but only triggers the debugger in case `trigger_gdb()` was called.
 * Tip: Call `trigger_gdb()` on your language of choice prior to any C API call to kickstart the debugger
 * on that LightGBM C API call.
 */
void gdb_on_trigger();



#endif   // LightGBM_UTILS_GDB_H_
