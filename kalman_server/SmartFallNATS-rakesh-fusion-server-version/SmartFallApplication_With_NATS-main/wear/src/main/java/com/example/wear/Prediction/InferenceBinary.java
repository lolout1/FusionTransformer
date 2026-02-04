/**
 * Encodes a single 128-sample accelerometer + gyroscope window into a compact,
 * versioned binary payload suitable for low-latency transmission (e.g., over NATS).
 *
 * <p><b>Payload layout (big-endian):</b></p>
 *
 * <pre>
 * [0–3]   Magic bytes        : 'S' 'F' 'N' '1'
 * [4]     Version            : 1
 * [5]     Flags              : bit0 = acc int16 scaled
 *                              bit1 = gyro int16 scaled
 * [6–7]   Reserved           : 0
 * [8–15]  Timestamp          : int64 (milliseconds since epoch)
 * [16–19] Sampling frequency : float32 (Hz)
 * [20]    UUID length        : uint8 (max 255)
 * [21..]  UUID bytes         : UTF-8 encoded
 *
 * Body:
 *   Accelerometer samples (128 × 3 × int16)
 *     Order per timestep: [ax, ay, az]
 *     Units: m/s² scaled by {@value #SCALE}
 *
 *   Gyroscope samples (128 × 3 × int16)
 *     Order per timestep: [gx, gy, gz]
 *     Units: rad/s scaled by {@value #SCALE}
 * </pre>
 *
 * <p><b>Numeric encoding:</b></p>
 * <ul>
 *   <li>Sensor values are multiplied by {@value #SCALE} and rounded to int16.</li>
 *   <li>Values exceeding int16 range are clamped to [-32768, 32767].</li>
 * </ul>
 *
 * <p><b>Constraints:</b></p>
 * <ul>
 *   <li>{@code acc128x3} and {@code gyro128x3} must be exactly [128][3].</li>
 *   <li>{@code uuid} is UTF-8 encoded and truncated to 255 bytes if longer.</li>
 * </ul>
 *
 * <p><b>Performance notes:</b></p>
 * <ul>
 *   <li>Uses a thread-local {@link ByteBuffer} to avoid per-call heap allocation.</li>
 *   <li>The returned byte array is a copy of the written region to ensure thread safety.</li>
 * </ul>
 *
 * @param uuid        Unique device or session identifier (UTF-8, max 255 bytes)
 * @param tsMillis    Timestamp of the window in milliseconds since epoch
 * @param fsHz        Sampling frequency in Hertz
 * @param acc128x3    Accelerometer window [128][3] in m/s²
 * @param gyro128x3   Gyroscope window [128][3] in rad/s
 *
 * @return Binary-encoded payload ready for transport or storage
 *
 * @throws NullPointerException if any argument is {@code null}
 * @throws ArrayIndexOutOfBoundsException if input arrays are not [128][3]
 */

package com.example.wear.Prediction;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

public final class InferenceBinary {
    private static final byte[] MAGIC = new byte[]{'S','F','N','1'};
    private static final byte VERSION = 1;
    private static final float SCALE = 1000.0f;

    // 128*3 per sensor
    private static final int COUNT = 128 * 3;

    // Header: 4+1+1+2+8+4+1 + uuidLen
    private static final int FIXED_HEADER_BYTES = 21;

    // Thread-local reusable buffer to avoid per-call allocation
    private static final ThreadLocal<ByteBuffer> TL_BUFFER = new ThreadLocal<>();

    private InferenceBinary() {}

    private static short clampToI16(int v) {
        if (v > 32767) return 32767;
        if (v < -32768) return -32768;
        return (short) v;
    }

    private static ByteBuffer getBuffer(int sizeBytes) {
        ByteBuffer bb = TL_BUFFER.get();
        if (bb == null || bb.capacity() < sizeBytes) {
            bb = ByteBuffer.allocate(sizeBytes).order(ByteOrder.BIG_ENDIAN);
            TL_BUFFER.set(bb);
        }
        bb.clear();
        bb.order(ByteOrder.BIG_ENDIAN);
        return bb;
    }

    public static byte[] encode(
            String uuid,
            long tsMillis,
            float fsHz,
            float[][] acc128x3,
            float[][] gyro128x3
    ) {
        byte[] uuidBytes = uuid.getBytes(StandardCharsets.UTF_8);
        int uuidLen = Math.min(uuidBytes.length, 255);

        int headerBytes = FIXED_HEADER_BYTES + uuidLen;
        int bodyBytes = (COUNT + COUNT) * 2; // acc + gyro int16
        int totalBytes = headerBytes + bodyBytes;

        // Reuse buffer (no per-call ByteBuffer.allocate)
        ByteBuffer bb = getBuffer(totalBytes);

        // magic + version + flags + reserved
        bb.put(MAGIC);
        bb.put(VERSION);

        byte flags = 0;
        flags |= 1; // bit0: acc int16 scaled
        flags |= 2; // bit1: gyro int16 scaled
        bb.put(flags);

        bb.putShort((short) 0); // reserved

        bb.putLong(tsMillis);
        bb.putFloat(fsHz);

        bb.put((byte) uuidLen);
        bb.put(uuidBytes, 0, uuidLen);

        // acc int16
        for (int t = 0; t < 128; t++) {
            // scale m/s^2 * 1000
            bb.putShort(clampToI16(Math.round(acc128x3[t][0] * SCALE)));
            bb.putShort(clampToI16(Math.round(acc128x3[t][1] * SCALE)));
            bb.putShort(clampToI16(Math.round(acc128x3[t][2] * SCALE)));
        }

        // gyro int16
        for (int t = 0; t < 128; t++) {
            // scale rad/s * 1000
            bb.putShort(clampToI16(Math.round(gyro128x3[t][0] * SCALE)));
            bb.putShort(clampToI16(Math.round(gyro128x3[t][1] * SCALE)));
            bb.putShort(clampToI16(Math.round(gyro128x3[t][2] * SCALE)));
        }

        // IMPORTANT: copy only the bytes we wrote, because bb's backing array is reused
        byte[] out = new byte[totalBytes];
        bb.flip();
        bb.get(out);
        return out;
    }
}