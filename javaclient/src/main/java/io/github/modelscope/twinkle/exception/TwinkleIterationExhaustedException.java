package io.github.modelscope.twinkle.exception;

import java.net.URI;

/** Indicates that a remote iterator was exhausted with HTTP status 410. */
public final class TwinkleIterationExhaustedException extends TwinkleServiceException {

    public TwinkleIterationExhaustedException(URI endpoint, String requestId, String detail) {
        super(410, endpoint, requestId, detail);
    }
}
