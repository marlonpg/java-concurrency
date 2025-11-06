package com.gamba.software.concurrency;

/**
 * Demonstrates Scoped Values for implicit context passing.
 * Scoped Values provide thread-local context without explicit parameter passing.
 */
public class ScopedValueExample {
    // Scoped values for implicit context - available to all methods in scope
    private static final ScopedValue<String> USER_ID = ScopedValue.newInstance();
    private static final ScopedValue<String> REQUEST_ID = ScopedValue.newInstance();
    
    /**
     * Sets up scoped context and processes request.
     * Values are automatically available to all nested method calls.
     */
    public void handleRequest(String userId, String requestId) {
        ScopedValue.where(USER_ID, userId)      // Bind USER_ID to scope
                  .where(REQUEST_ID, requestId) // Bind REQUEST_ID to scope
                  .run(() -> {
                      processRequest(); // All nested calls can access scoped values
                      // USER_ID and REQUEST_ID are available in all called methods
                  });
        // Values automatically cleaned up here - no manual cleanup needed
    }
    
    /**
     * Processes request using scoped values - no parameters needed.
     */
    private void processRequest() {
        String currentUser = USER_ID.get();    // Available without passing parameters
        String currentRequest = REQUEST_ID.get(); // Implicit context access
        
        // Call other methods that can also access these values
        validateUser(); // Can access USER_ID without parameters
        logRequest();   // Can access both values without parameters
    }
    
    /**
     * Validates user using scoped context - no parameters required.
     */
    private void validateUser() {
        String userId = USER_ID.get(); // Still available in nested calls
        // Validation logic here
        System.out.println("Validating user: " + userId);
    }
    
    /**
     * Logs request details using both scoped values.
     */
    private void logRequest() {
        String userId = USER_ID.get();    // Access user context
        String requestId = REQUEST_ID.get(); // Access request context
        System.out.println("Processing request " + requestId + " for user " + userId);
    }
}