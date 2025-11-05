package com.gamba.software.concurrency;

import java.util.concurrent.ExecutionException;

public class ConcurrencyExamplesRunner {
    
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        System.out.println("=== Java Concurrency Examples ===\n");
        
        // CompletableFuture Example
        System.out.println("1. CompletableFuture Example:");
        CompletableFutureAdvanced cfExample = new CompletableFutureAdvanced();
        String result = cfExample.processOrder("ORDER-123").get();
        System.out.println("Final result: " + result);
        cfExample.shutdown();
        System.out.println();
        
        // VarHandle Example
        System.out.println("2. VarHandle Example:");
        VarHandleExample varExample = new VarHandleExample();
        System.out.println("Initial value: " + varExample.get());
        System.out.println("Increment: " + varExample.getAndIncrement());
        System.out.println("Current value: " + varExample.get());
        System.out.println("CAS (1 -> 5): " + varExample.compareAndSet(1, 5));
        System.out.println("Final value: " + varExample.get());
        System.out.println();
        
        // Structured Concurrency Example
        System.out.println("3. Structured Concurrency Example:");
        StructuredConcurrencyExample scExample = new StructuredConcurrencyExample();
        String scResult = scExample.handleRequest("REQ-456");
        System.out.println("Structured result: " + scResult);
        System.out.println();
        
        // Scoped Values Example
        System.out.println("4. Scoped Values Example:");
        ScopedValueExample svExample = new ScopedValueExample();
        svExample.handleRequest("user123", "request789");
        System.out.println();
        
        // Lock-Free Stack Example
        System.out.println("5. Lock-Free Stack Example:");
        LockFreeStack<String> stack = new LockFreeStack<>();
        stack.push("First");
        stack.push("Second");
        stack.push("Third");
        
        System.out.println("Popping from stack:");
        while (!stack.isEmpty()) {
            System.out.println("Popped: " + stack.pop());
        }
        
        System.out.println("\n=== All examples completed ===");
    }
}