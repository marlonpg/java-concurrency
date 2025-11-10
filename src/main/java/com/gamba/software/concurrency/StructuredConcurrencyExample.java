package com.gamba.software.concurrency;

/**
 * Demonstrates Structured Concurrency for coordinated parallel task execution.
 * Ensures all related tasks are properly managed and cleaned up together.
 */
public class StructuredConcurrencyExample {
//    /**
//     * Handles request by executing multiple steps in parallel.
//     * Uses structured concurrency to ensure proper task lifecycle management.
//     */
//    public String handleRequest(String requestId) throws InterruptedException, ExecutionException {
//        try (var scope = new StructuredTaskScope.ShutdownOnFailure()) { // Auto-cleanup on failure
//            // Fork multiple related tasks - all execute in parallel
//            Supplier<String> task1 = scope.fork(() -> processStep1(requestId));
//            Supplier<String> task2 = scope.fork(() -> processStep2(requestId));
//            Supplier<String> task3 = scope.fork(() -> processStep3(requestId));
//
//            scope.join();           // Wait for all tasks to complete
//            scope.throwIfFailed();  // Propagate any failures from child tasks
//
//            // All tasks succeeded - combine results
//            return combineResults(
//                task1.get(), // Safe to call - all tasks completed successfully
//                task2.get(),
//                task3.get()
//            );
//        } // Automatic cleanup of any remaining tasks - structured guarantee
//    }
//
//    // Simulates first processing step with 100ms delay
//    private String processStep1(String requestId) {
//        try { Thread.sleep(100); } catch (InterruptedException e) {} // Simulate work
//        return "Step1-" + requestId;
//    }
//
//    // Simulates second processing step with 150ms delay
//    private String processStep2(String requestId) {
//        try { Thread.sleep(150); } catch (InterruptedException e) {} // Simulate work
//        return "Step2-" + requestId;
//    }
//
//    // Simulates third processing step with 200ms delay
//    private String processStep3(String requestId) {
//        try { Thread.sleep(200); } catch (InterruptedException e) {} // Simulate work
//        return "Step3-" + requestId;
//    }
//
//    // Combines results from all parallel steps into final result
//    private String combineResults(String s1, String s2, String s3) {
//        return String.format("Combined: %s, %s, %s", s1, s2, s3);
//    }
}