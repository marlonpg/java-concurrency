package com.gamba.software.concurrency;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Demonstrates which AtomicInteger methods do NOT use CAS operations.
 * These methods use different mechanisms for thread safety.
 */
public class NonCASMethods {
    
    public static void main(String[] args) {
        demonstrateNonCASMethods();
        demonstrateVolatileOperations();
    }
    
    private static void demonstrateNonCASMethods() {
        System.out.println("=== Methods that do NOT use CAS ===");
        AtomicInteger atomic = new AtomicInteger(42);
        
        // 1. get() - Simple volatile read, no CAS needed
        System.out.println("get(): " + atomic.get() + " (volatile read only)");
        
        // 2. set() - Simple volatile write, no CAS needed  
        atomic.set(100);
        System.out.println("set(100): " + atomic.get() + " (volatile write only)");
        
        // 3. lazySet() - Ordered write, weaker than volatile, no CAS
        atomic.lazySet(200);
        System.out.println("lazySet(200): " + atomic.get() + " (ordered write, no immediate visibility guarantee)");
        
        System.out.println("\n=== Methods that DO use CAS ===");
        
        // These methods use CAS internally:
        System.out.println("incrementAndGet(): " + atomic.incrementAndGet() + " (uses CAS loop)");
        System.out.println("addAndGet(5): " + atomic.addAndGet(5) + " (uses CAS loop)");
        System.out.println("compareAndSet(206, 300): " + atomic.compareAndSet(206, 300) + " (explicit CAS)");
    }
    
    private static void demonstrateVolatileOperations() {
        System.out.println("\n=== Volatile vs CAS Operations ===");
        
        AtomicReference<String> atomicRef = new AtomicReference<>("initial");
        
        // Non-CAS operations:
        System.out.println("get(): " + atomicRef.get() + " (volatile read)");
        atomicRef.set("updated");
        System.out.println("set(): " + atomicRef.get() + " (volatile write)");
        
        // CAS operations:
        boolean success = atomicRef.compareAndSet("updated", "final");
        System.out.println("compareAndSet(): " + success + ", value: " + atomicRef.get() + " (CAS operation)");
    }
}