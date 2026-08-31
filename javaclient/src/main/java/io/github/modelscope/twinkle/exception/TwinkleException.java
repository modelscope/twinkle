package io.github.modelscope.twinkle.exception;

/** Serves as the base class for all Twinkle client runtime exceptions. */
public class TwinkleException extends RuntimeException {

    public TwinkleException(String message) {
        super(message);
    }

    public TwinkleException(String message, Throwable cause) {
        super(message, cause);
    }
}
