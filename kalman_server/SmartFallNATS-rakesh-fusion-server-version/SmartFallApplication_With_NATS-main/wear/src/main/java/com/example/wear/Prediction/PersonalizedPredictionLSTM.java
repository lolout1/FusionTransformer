package com.example.wear.Prediction;

import android.content.Context;
import android.content.Intent;
import android.util.Log;
import android.os.SystemClock;

import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import com.example.wear.MainActivity;
import com.example.wear.NatsManager;
import com.example.wear.config.ModelConfig;

import com.google.gson.Gson;

import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.io.FileOutputStream;

import io.nats.client.Connection;
import io.nats.client.Message;


public class PersonalizedPredictionLSTM {

    private static final String TAG = "PersonalizedPredictionLSTM";
    private static float[] MODEL_THRESHOLDS = {0.70f};

    public static Context con;

    // -----------------------------
    // Simple backpressure on watch
    // -----------------------------
    private static final Object IN_FLIGHT_LOCK = new Object();
    private static boolean IN_FLIGHT = false;

    // Timeouts
    private static final long REQUEST_TIMEOUT_MS = 500; // 250-500ms recommended
    private static final long MIN_REQUEST_GAP_MS = 20;   // set 20-100ms if want throttling
    private static volatile long lastRequestElapsedMs = 0L;

    /**
     * Initializes context
     */
    public static void initialize(Context context) {
        con = context;
    }

    // -----------------------------------------------------------------------
    // Container for raw window + low-movement decision
    // (We compute magnitude only for early-exit; we DO NOT send mag as a channel)
    // -----------------------------------------------------------------------
    private static class WindowXYZ {
        float[][] xyz;         // shape: [time][3] -> x,y,z (raw)
        boolean isLowMovement; // true if all magnitudes <= threshold
    }

    // -----------------------------------------------------------------------
    // JSON payload for server-side Kalman fusion + model inference
    // -----------------------------------------------------------------------
    private static class InferencePayload {
        String uuid;
        long tsMillis;
        float fsHz;

        float[][] acc;    // [128][3] raw accel
        float[][] gyro;   // [128][3] raw gyro

        String unitsAcc;   // "m/s^2"
        String unitsGyro;  // "rad/s"
    }

    // -----------------------------------------------------------------------
    // Compute magnitude ONCE for early exit, but keep raw xyz for sending
    // -----------------------------------------------------------------------
    private static WindowXYZ checkMovementAndKeepXYZ(float[][] window, float threshold) {
        int timeSteps = window.length;

        WindowXYZ result = new WindowXYZ();
        result.xyz = new float[timeSteps][3];

        boolean lowMovement = true;

        for (int t = 0; t < timeSteps; t++) {
            float x = window[t][0];
            float y = window[t][1];
            float z = window[t][2];

            result.xyz[t][0] = x;
            result.xyz[t][1] = y;
            result.xyz[t][2] = z;

            float mag = (float) Math.sqrt(x * x + y * y + z * z);
            if (mag > threshold) {
                lowMovement = false;
            }
        }

        result.isLowMovement = lowMovement;
        return result;
    }

    /**
     * Sends raw accelerometer + gyroscope windows to server.
     * Server performs Kalman fusion and runs the final transformer model (best_model.pth).
     *
     * @param accWindow  float[128][3] -> ax,ay,az
     * @param gyroWindow float[128][3] -> gx,gy,gz
     * @return inference probability (float), or 0.0 on failure/timeout
     */
    public static float makeInference(float[][] accWindow, float[][] gyroWindow) throws Exception {
        float inference = 0.0f;

        // ---------------------------------------------------------
        // 0) Basic shape validation (avoid silent errors)
        // ---------------------------------------------------------
        if (accWindow == null || gyroWindow == null) {
            Log.e(TAG, "accWindow or gyroWindow is null");
            return 0.0f;
        }
        if (accWindow.length != gyroWindow.length) {
            Log.e(TAG, "accWindow and gyroWindow length mismatch: "
                    + accWindow.length + " vs " + gyroWindow.length);
            return 0.0f;
        }
        if (accWindow.length < 128) {
            Log.e(TAG, "Need at least 128 samples. Got: " + accWindow.length);
            return 0.0f;
        }
        if (accWindow[0].length < 3 || gyroWindow[0].length < 3) {
            Log.e(TAG, "Expected [T][3] arrays for acc & gyro.");
            return 0.0f;
        }

        // ---------------------------------------------------------
        // 1) Sensor-specific low-movement thresholds
        // Set these conservatively so it don't skip true falls.
        // ---------------------------------------------------------
        final float ACC_THRESH  = 20.0f;  // if acc in g: use ~2.0f
        final float GYRO_THRESH = 5.0f;   // if gyro in rad/s: use ~300 in deg/s

        // ---------------------------------------------------------
        // 2) Single-pass check per sensor; keep raw xyz
        // ---------------------------------------------------------
        WindowXYZ accData  = checkMovementAndKeepXYZ(accWindow,  ACC_THRESH);
        WindowXYZ gyroData = checkMovementAndKeepXYZ(gyroWindow, GYRO_THRESH);

        // ---------------------------------------------------------
        // 3) Early exit logic (skip server call)
        // ---------------------------------------------------------
        if (accData.isLowMovement && gyroData.isLowMovement) {
            Log.d(TAG, "Bypassed server inference → Low movement (ACC+GYRO)");
            return 0.02f;
        }

        // ---------------------------------------------------------
        // 4) Watch status + UUID
        // ---------------------------------------------------------
        String status = MainActivity.watchStatus;
        if (!"Activated".equals(status)) {
            return 0.0f; // user requested: no stale / cached prob
        }

        ModelConfig config = ModelConfig.getModelConfig(con);
        String uuid = (config != null) ? config.uuid : null;
        if (uuid == null || uuid.isEmpty()) {
            Log.e(TAG, "UUID missing in ModelConfig");
            return 0.0f;
        }

        // Optional throttle (battery + server load) - to skip few server infernces
        long nowElapsed = SystemClock.elapsedRealtime();
        if (MIN_REQUEST_GAP_MS > 0 && (nowElapsed - lastRequestElapsedMs) < MIN_REQUEST_GAP_MS) {
            return 0.0f;
        }

        // In-flight backpressure: do not queue requests
        synchronized (IN_FLIGHT_LOCK) {
            if (IN_FLIGHT) {
                // We purposely avoid building up backlog on the watch
                return 0.0f;
            }
            IN_FLIGHT = true;
        }

        try {
            // NATS subject
            String subject = "m.kalman_transformer." + uuid;

            InferencePayload payload = new InferencePayload();
            payload.uuid = uuid;
            payload.tsMillis = System.currentTimeMillis();
            payload.fsHz = 30.0f;
            payload.acc = sliceFirst128(accData.xyz);
            payload.gyro = sliceFirst128(gyroData.xyz);

            // IMPORTANT: set correct units for sensors:
            payload.unitsAcc = "m/s^2";    // TYPE_ACCELEROMETER reports m/s² and includes gravity.
            payload.unitsGyro = "rad/s";   // TYPE_GYROSCOPE reports rad/s.

            byte[] data = InferenceBinary.encode(
                    uuid,
                    payload.tsMillis,
                    payload.fsHz,
                    payload.acc,
                    payload.gyro
            );

            // ---------------------------------------------------------
            // 5) Send request to server and parse float response
            // ---------------------------------------------------------
            long t0Elapsed = SystemClock.elapsedRealtime();

            // Ensure connection
            Connection c = NatsManager.getConnectionOrNull();
            if (c == null && !NatsManager.awaitConnected(1500)) {
                Log.v(TAG, "NATS connection is null (after await). Skipping cycle.");
                return 0.0f;
            }
            c = NatsManager.getConnectionOrNull();
            if (c == null) {
                Log.v(TAG, "NATS connection still null. Skipping cycle.");
                return 0.0f;
            }

            CompletableFuture<Message> future = c.request(subject, data);

            // Tight timeout
            Message m = future.get(REQUEST_TIMEOUT_MS, TimeUnit.MILLISECONDS);

            String resp = new String(m.getData()).trim();
            inference = Float.parseFloat(resp);

            long t1Elapsed = SystemClock.elapsedRealtime();
            long rttMs = t1Elapsed - t0Elapsed;

            lastRequestElapsedMs = t0Elapsed;

            Log.d(TAG, "Score: " + inference + " | NATS request duration: " + rttMs + " ms");

        } catch (NumberFormatException nfe) {
            Log.e(TAG, "Bad score payload (non-float): " + nfe.getMessage());
            inference = 0.0f;
        } catch (TimeoutException e) {
            Log.e(TAG, "NATS response timeout: " + e.getMessage());
            inference = 0.0f;
        } catch (CancellationException e) {
            Log.e(TAG, "Request cancelled (no responders?): " + e.getMessage());
            inference = 0.0f;
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            Log.d(TAG, "Inference interrupted (watch deactivated / service stopping).");
            return 0.0f;
        } catch (ExecutionException e) {
            Log.e(TAG, "Execution exception: " + e.getMessage());
            inference = 0.0f;
        } finally {
            synchronized (IN_FLIGHT_LOCK) {
                IN_FLIGHT = false;
            }
        }

        return inference;
    }

    // -----------------------------------------------------------------------
    // Utility: ensure we send exactly 128 samples
    // -----------------------------------------------------------------------
    private static float[][] sliceFirst128(float[][] xyz) {
        int T = 128;
        float[][] out = new float[T][3];
        for (int i = 0; i < T; i++) {
            out[i][0] = xyz[i][0];
            out[i][1] = xyz[i][1];
            out[i][2] = xyz[i][2];
        }
        return out;
    }

    public static float getThreshold() {
        return MODEL_THRESHOLDS[0];
    }

    /**
     * This method serializes the ModelConfig object and writes the result to SmartWatchValues.json
     * @param context
     */
    public static void updateConfigFile(Context context){
        ModelConfig config = ModelConfig.getModelConfig(context);
        String filename =  "SmartWatchValues.json" ;

        Gson gson = new Gson();
        String jsonString = gson.toJson(config);

        Intent modelIntent = new Intent("modelInfo");
        modelIntent.putExtra("data", jsonString);
        LocalBroadcastManager.getInstance(context).sendBroadcast(modelIntent);

        try (FileOutputStream fos = context.openFileOutput(filename, Context.MODE_PRIVATE)) {
            fos.write(jsonString.getBytes());
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

}