package com.gamba.software.concurrency;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

/**
 * Demonstrates VarHandle API for low-level atomic operations.
 * VarHandle provides fine-grained control over memory access patterns.
 */
public class VarHandleExample {
    // VarHandle for atomic operations on counter field
    private static final VarHandle COUNTER;
    private volatile int counter; // Target field for VarHandle operations
    
    // Static initialization of VarHandle - must be done at class loading time
    static {
        try {
            // Create VarHandle for the 'counter' field of this class
            COUNTER = MethodHandles.lookup()
                .findVarHandle(VarHandleExample.class, "counter", int.class);
        } catch (Exception e) {
            throw new RuntimeException(e); // VarHandle creation failed
        }
    }
    
    /**
     * Atomic compare-and-set operation using VarHandle.
     * Updates value only if current value matches expected.
     */
    public boolean compareAndSet(int expected, int update) {
        return COUNTER.compareAndSet(this, expected, update);
    }
    
    /**
     * Atomic get-and-increment operation.
     * Returns old value and increments by 1.
     */
    public int getAndIncrement() {
        return (int) COUNTER.getAndAdd(this, 1); // Atomic add operation
    }
    
    /**
     * Sets value using opaque access mode.
     * Provides atomicity but no ordering guarantees.
     */
    public void setOpaque(int value) {
        COUNTER.setOpaque(this, value); // Opaque access mode - weaker than volatile
    }
    
    /**
     * Gets value using acquire access mode.
     * Provides acquire semantics for memory ordering.
     */
    public int getAcquire() {
        return (int) COUNTER.getAcquire(this); // Acquire access mode - one-way memory barrier
    }
    
    /**
     * Simple volatile read of counter value.
     */
    public int get() {
        return counter; // Standard volatile read
    }
}