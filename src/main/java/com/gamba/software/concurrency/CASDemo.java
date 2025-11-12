package com.gamba.software.concurrency;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Demonstrates that AtomicInteger uses Compare-and-Swap (CAS) operations.
 * CAS is a lock-free atomic operation that updates a value only if it matches an expected value.
 */
public class CASDemo {
    
    public static void main(String[] args) {
        demonstrateCAS();
        demonstrateInternalCAS();
    }
    
    /**
     * Shows explicit CAS usage - the foundation of all atomic operations
     */
    private static void demonstrateCAS() {
        System.out.println("=== Explicit CAS Operations ===");
        AtomicInteger atomic = new AtomicInteger(10);
        
        // CAS: Compare current value (10) with expected (10), if match set to 20
        boolean success1 = atomic.compareAndSet(10, 20);
        System.out.println("CAS(10->20): " + success1 + ", value: " + atomic.get());
        
        // CAS: Compare current value (20) with expected (10), should fail
        boolean success2 = atomic.compareAndSet(10, 30);
        System.out.println("CAS(10->30): " + success2 + ", value: " + atomic.get());
        
        // CAS: Compare current value (20) with expected (20), should succeed
        boolean success3 = atomic.compareAndSet(20, 30);
        System.out.println("CAS(20->30): " + success3 + ", value: " + atomic.get());
    }
    
    /**
     * Shows how increment operations use CAS internally
     */
    private static void demonstrateInternalCAS() {
        System.out.println("\n=== How incrementAndGet() uses CAS internally ===");
        AtomicInteger atomic = new AtomicInteger(0);
        
        // This is essentially what incrementAndGet() does internally:
        int current, next;
        do {
            current = atomic.get();           // Read current value
            next = current + 1;               // Calculate new value
            System.out.println("Attempting CAS: " + current + " -> " + next);
        } while (!atomic.compareAndSet(current, next)); // Retry until CAS succeeds
        
        System.out.println("Final value: " + atomic.get());
        
        // Compare with built-in method
        System.out.println("Built-in incrementAndGet(): " + atomic.incrementAndGet());
    }
}