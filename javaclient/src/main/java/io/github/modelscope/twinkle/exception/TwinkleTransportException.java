package io.github.modelscope.twinkle.exception;

/** Indicates a local HTTP transport, connectivity, or timeout failure. */
public final class TwinkleTransportException extends TwinkleException {

    public TwinkleTransportException(String message, Throwable cause) {
        super(message, cause);
    }
}
