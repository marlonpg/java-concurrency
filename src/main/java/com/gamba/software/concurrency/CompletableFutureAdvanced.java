package com.gamba.software.concurrency;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Advanced CompletableFuture example demonstrating asynchronous order processing.
 * Shows chaining of async operations, error handling, and resource management.
 */
public class CompletableFutureAdvanced {
    // Thread pool for executing async tasks - fixed size for predictable resource usage
    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    
    /**
     * Processes an order through multiple async stages.
     * Demonstrates method chaining, composition, and error handling.
     */
    public CompletableFuture<String> processOrder(String orderId) {
        return CompletableFuture
            // Start async validation
            .supplyAsync(() -> validateOrder(orderId), executor)
            // Chain inventory fetch (returns CompletableFuture)
            .thenCompose(this::fetchInventory)
            // Chain price calculation (returns CompletableFuture)
            .thenCompose(this::calculatePrice)
            // Transform final result (synchronous)
            .thenApply(this::formatResult)
            // Handle any exceptions in the chain
            .exceptionally(throwable -> "Order failed: " + throwable.getMessage())
            // Log completion regardless of success/failure
            .whenComplete((result, throwable) -> {
                if (throwable == null) {
                    System.out.println("Order completed: " + result);
                } else {
                    System.err.println("Order failed: " + throwable);
                }
            });
    }
    
    // Validates order ID - throws exception if invalid
    private String validateOrder(String orderId) {
        if (orderId == null) throw new IllegalArgumentException("Invalid order");
        return orderId;
    }
    
    // Simulates async inventory lookup - returns CompletableFuture for chaining
    private CompletableFuture<String> fetchInventory(String orderId) {
        return CompletableFuture.supplyAsync(() -> orderId + "-inventory", executor);
    }
    
    // Simulates async price calculation based on inventory
    private CompletableFuture<String> calculatePrice(String inventory) {
        return CompletableFuture.supplyAsync(() -> inventory + "-$100", executor);
    }
    
    // Final formatting step - synchronous transformation
    private String formatResult(String priceInfo) {
        return "Processed: " + priceInfo;
    }
    
    // Clean shutdown of thread pool - important for resource cleanup
    public void shutdown() {
        executor.shutdown();
    }
}