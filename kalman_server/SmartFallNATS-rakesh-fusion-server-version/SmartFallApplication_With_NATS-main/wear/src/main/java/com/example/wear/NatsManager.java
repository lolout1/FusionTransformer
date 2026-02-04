package com.example.wear;

import android.content.Context;
import android.util.Log;

import com.example.wear.util.SSLUtils;

import java.time.Duration;
import java.util.concurrent.Executors;

import io.nats.client.Connection;
import io.nats.client.Nats;
import io.nats.client.Options;

import javax.net.ssl.SSLContext;

public class NatsManager {

    private static final String TAG = "Nats Service";

    // Make nc visible across threads, and avoid stale reads
    private static volatile Connection nc;
    public static volatile boolean connect = false;

    private final MainActivity datacollector;
    private final Context context;

    public NatsManager(MainActivity datacollector, Context context) {
        this.datacollector = datacollector;
        this.context = context;
    }

    public static void pub(String topic, String data) {
        Connection c = nc;
        if (connect && c != null && c.getStatus() == Connection.Status.CONNECTED) {
            c.publish(topic, data.getBytes());
        } else {
            Log.w(TAG, "Publish skipped: NATS not connected");
        }
    }

    /** Non-blocking connect kickoff */
    public void connect() {
        Log.d(TAG, "TRY TO CONNECT");
        new Thread(() -> {
    
            // Build two option sets: TLS and plaintext.
            Options tlsOptions = null;
            try {
                SSLContext sslContext = SSLUtils.createInsecureSSLContext(); // your helper
                tlsOptions = new Options.Builder()
                        .server("tls://chocolatefrog@cssmartfall1.cose.txstate.edu:4224")
                        .sslContext(sslContext)
                        .maxReconnects(-1)
                        .reconnectWait(java.time.Duration.ofMillis(500))
                        .connectionTimeout(java.time.Duration.ofSeconds(3))
                        .connectionListener((conn, type) -> {
                            Log.i(TAG, "NATS event: " + type);
                            switch (type) {
                                case CONNECTED:
                                case RECONNECTED: connect = true; break;
                                case DISCONNECTED:
                                case CLOSED:      connect = false; break;
                                default: break;
                            }
                        })
                        .errorListener(new io.nats.client.ErrorListener() {
                            @Override public void exceptionOccurred(io.nats.client.Connection conn, Exception exp) {
                                Log.w(TAG, "NATS exception: " + exp);
                            }
                            @Override public void errorOccurred(io.nats.client.Connection conn, String type) {
                                Log.w(TAG, "NATS error: " + type);
                            }
                            @Override public void slowConsumerDetected(io.nats.client.Connection conn, io.nats.client.Consumer consumer) {
                                Log.w(TAG, "NATS slow consumer detected");
                            }
                        })
                        .build();
            } catch (Exception e) {
                Log.w(TAG, "Could not build TLS options, will try plaintext.", e);
            }
    
            Options plainOptions = new Options.Builder()
                    // IMPORTANT: plaintext scheme
                    .server("nats://chocolatefrog@cssmartfall1.cose.txstate.edu:4224")
                    .maxReconnects(-1)
                    .reconnectWait(java.time.Duration.ofMillis(500))
                    .connectionTimeout(java.time.Duration.ofSeconds(3))
                    .connectionListener((conn, type) -> {
                        Log.i(TAG, "NATS event: " + type);
                        switch (type) {
                            case CONNECTED:
                            case RECONNECTED: connect = true; break;
                            case DISCONNECTED:
                            case CLOSED:      connect = false; break;
                            default: break;
                        }
                    })
                    .errorListener(new io.nats.client.ErrorListener() {
                        @Override public void exceptionOccurred(io.nats.client.Connection conn, Exception exp) {
                            Log.w(TAG, "NATS exception: " + exp);
                        }
                        @Override public void errorOccurred(io.nats.client.Connection conn, String type) {
                            Log.w(TAG, "NATS error: " + type);
                        }
                        @Override public void slowConsumerDetected(io.nats.client.Connection conn, io.nats.client.Consumer consumer) {
                            Log.w(TAG, "NATS slow consumer detected");
                        }
                    })
                    .build();
    
            // Try TLS first (if configured), then fallback to plaintext.
            try {
                if (tlsOptions != null) {
                    Log.i(TAG, "Attempting TLS connect on 4224...");
                    nc = Nats.connect(tlsOptions);
                    Log.i(TAG, "TLS connection established.");
                    return; // success
                }
            } catch (Exception tlsExp) {
                Log.w(TAG, "TLS connect failed, falling back to plaintext: " + tlsExp);
                nc = null;
                connect = false;
            }
    
            try {
                Log.i(TAG, "Attempting PLAINTEXT connect on 4224...");
                nc = Nats.connect(plainOptions);
                Log.i(TAG, "Plaintext connection established.");
            } catch (Exception exp) {
                Log.e(TAG, "Failed to connect to NATS server (both TLS and plaintext)", exp);
                nc = null;
                connect = false;
            }
        }).start();
    }
    
    /** Graceful close (null-safe). */
    public void close() {
        try {
            Connection c = nc;
            if (c != null) {
                c.close();
                Log.d(TAG, "Nats connection closed");
            }
        } catch (InterruptedException e) {
            Log.e(TAG, "Error closing Nats connection", e);
            Thread.currentThread().interrupt();
        } finally {
            nc = null;
            connect = false;
        }
    }

    // ----------------------
    // NEW: tiny safety helpers
    // ----------------------

    /** Returns true if connected right now. */
    public static boolean isConnected() {
        Connection c = nc;
        return c != null && c.getStatus() == Connection.Status.CONNECTED;
    }

    /**
     * Best-effort wait for an active connection.
     * Polls the connection state for up to timeoutMs and returns true if connected.
     */
    public static boolean awaitConnected(long timeoutMs) {
        long end = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < end) {
            if (isConnected()) return true;
            try { Thread.sleep(50); } catch (InterruptedException ignored) { }
        }
        return isConnected();
    }

    /**
     * Returns the current Connection or null if not connected.
     * Use this before calling request(...) to avoid NPE.
     */
    public static Connection getConnectionOrNull() {
        return isConnected() ? nc : null;
    }
}