package com.gamba.software.concurrency;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CompletableFutureAdvanced {
    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    
    public CompletableFuture<String> processOrder(String orderId) {
        return CompletableFuture
            .supplyAsync(() -> validateOrder(orderId), executor)
            .thenCompose(this::fetchInventory)
            .thenCompose(this::calculatePrice)
            .thenApply(this::formatResult)
            .exceptionally(throwable -> "Order failed: " + throwable.getMessage())
            .whenComplete((result, throwable) -> {
                if (throwable == null) {
                    System.out.println("Order completed: " + result);
                } else {
                    System.err.println("Order failed: " + throwable);
                }
            });
    }
    
    private String validateOrder(String orderId) {
        if (orderId == null) throw new IllegalArgumentException("Invalid order");
        return orderId;
    }
    
    private CompletableFuture<String> fetchInventory(String orderId) {
        return CompletableFuture.supplyAsync(() -> orderId + "-inventory", executor);
    }
    
    private CompletableFuture<String> calculatePrice(String inventory) {
        return CompletableFuture.supplyAsync(() -> inventory + "-$100", executor);
    }
    
    private String formatResult(String priceInfo) {
        return "Processed: " + priceInfo;
    }
    
    public void shutdown() {
        executor.shutdown();
    }
}