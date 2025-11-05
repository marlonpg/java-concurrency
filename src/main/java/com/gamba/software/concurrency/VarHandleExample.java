package com.gamba.software.concurrency;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

public class VarHandleExample {
    private static final VarHandle COUNTER;
    private volatile int counter;
    
    static {
        try {
            COUNTER = MethodHandles.lookup()
                .findVarHandle(VarHandleExample.class, "counter", int.class);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    
    public boolean compareAndSet(int expected, int update) {
        return COUNTER.compareAndSet(this, expected, update);
    }
    
    public int getAndIncrement() {
        return (int) COUNTER.getAndAdd(this, 1);
    }
    
    public void setOpaque(int value) {
        COUNTER.setOpaque(this, value); // Opaque access mode
    }
    
    public int getAcquire() {
        return (int) COUNTER.getAcquire(this); // Acquire access mode
    }
    
    public int get() {
        return counter;
    }
}