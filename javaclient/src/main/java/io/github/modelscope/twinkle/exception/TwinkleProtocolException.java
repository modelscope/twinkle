package io.github.modelscope.twinkle.exception;

/** Indicates that a server response does not match the expected JSON protocol. */
public final class TwinkleProtocolException extends TwinkleException {

    public TwinkleProtocolException(String message, Throwable cause) {
        super(message, cause);
    }
}
