package com.dankook.mlpa_gradi.exception;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.context.request.async.AsyncRequestNotUsableException;
import jakarta.servlet.http.HttpServletRequest;

import java.util.Map;

@RestControllerAdvice(basePackages = "com.dankook.mlpa_gradi")
@Slf4j
public class GlobalExceptionHandler {

    @ExceptionHandler(AsyncRequestNotUsableException.class)
    public void handleAsyncNotUsable(AsyncRequestNotUsableException e) {
        log.debug("Async connection lost: {}", e.getMessage());
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<Map<String, Object>> handleAllExceptions(Exception e, HttpServletRequest request) {
        // SSE 요청인 경우 JSON 응답을 보내면 HttpMessageNotWritableException이 발생하므로 무시
        String accept = request.getHeader("Accept");
        if (accept != null && accept.contains("text/event-stream")) {
            log.warn("Error during SSE stream (Ignored JSON response): {}", e.getMessage());
            return null;
        }

        log.error("Unhandled Exception: ", e);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(Map.of(
                        "error", "Internal Server Error",
                        "message", e.getMessage() != null ? e.getMessage() : "Unknown Error"));
    }
}
