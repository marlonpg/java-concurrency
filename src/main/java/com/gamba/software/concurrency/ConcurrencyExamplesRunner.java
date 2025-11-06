package com.gamba.software.concurrency;

import java.util.concurrent.ExecutionException;

/**
 * Main runner class that demonstrates various Java concurrency features.
 * Executes examples of CompletableFuture, VarHandle, Structured Concurrency,
 * Scoped Values, and Lock-Free data structures.
 */
public class ConcurrencyExamplesRunner {
    
    /**
     * Main entry point - runs all concurrency examples in sequence.
     */
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        System.out.println("=== Java Concurrency Examples ===\n");
        
        // Demonstrates async processing with CompletableFuture chaining
        System.out.println("1. CompletableFuture Example:");
        CompletableFutureAdvanced cfExample = new CompletableFutureAdvanced();
        String result = cfExample.processOrder("ORDER-123").get(); // Blocking wait for result
        System.out.println("Final result: " + result);
        cfExample.shutdown(); // Clean up thread pool
        System.out.println();
        
        // Shows low-level atomic operations using VarHandle API
        System.out.println("2. VarHandle Example:");
        VarHandleExample varExample = new VarHandleExample();
        System.out.println("Initial value: " + varExample.get());
        System.out.println("Increment: " + varExample.getAndIncrement()); // Atomic increment
        System.out.println("Current value: " + varExample.get());
        System.out.println("CAS (1 -> 5): " + varExample.compareAndSet(1, 5)); // Compare-and-swap
        System.out.println("Final value: " + varExample.get());
        System.out.println();
        
        // Demonstrates structured concurrency for coordinated task execution
        System.out.println("3. Structured Concurrency Example:");
        StructuredConcurrencyExample scExample = new StructuredConcurrencyExample();
        String scResult = scExample.handleRequest("REQ-456"); // Parallel task execution
        System.out.println("Structured result: " + scResult);
        System.out.println();
        
        // Shows thread-local context passing without explicit parameters
        System.out.println("4. Scoped Values Example:");
        ScopedValueExample svExample = new ScopedValueExample();
        svExample.handleRequest("user123", "request789"); // Context automatically available
        System.out.println();
        
        // Demonstrates lock-free data structure using atomic operations
        System.out.println("5. Lock-Free Stack Example:");
        LockFreeStack<String> stack = new LockFreeStack<>();
        stack.push("First");   // Atomic push operations
        stack.push("Second");
        stack.push("Third");
        
        System.out.println("Popping from stack:");
        while (!stack.isEmpty()) {
            System.out.println("Popped: " + stack.pop()); // Atomic pop operations
        }
        
        System.out.println("\n=== All examples completed ===");
    }
}