package com.example.wear;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;
import android.util.TimeUtils;
import android.widget.TimePicker;

import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import com.example.wear.Prediction.Prediction;
import com.example.wear.util.Event;

import org.json.JSONException;
import org.json.JSONObject;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.sql.Timestamp;
import java.time.Instant;
import java.util.Timer;
import java.util.TimerTask;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SensorService extends Service implements SensorEventListener {

    private final String TAG = "Sensor Service";
    private SensorManager mmSensorManager;
    // private Sensor mmSensor;
    private Sensor accSensor;
    private Sensor gyroSensor;

    private float[] mmSensorValues;
    private Timer timer;

    private float start_time=0;

    private long startTime=0L;
    private long startTime1=0L;
    private long latency=0L;
    private long total=0L;
    private long endTime=0L;
    private long endTime1=0L;
    private int predictions = 0;
    String status=MainActivity.watchStatus;

    private static int noOfEvents;
    private float[] latestAcc = null;
    private float[] latestGyro = null;

    @Override
    public void onCreate() {
        super.onCreate();


            timer = new Timer();
            timer.scheduleAtFixedRate(new TimerTask() {
                @Override
                public void run() {
                    if (mmSensorValues != null) {
                        Intent messageIntent = new Intent("data-receiver");
                        messageIntent.putExtra("message", mmSensorValues[1] + "");
                        LocalBroadcastManager.getInstance(getApplicationContext()).sendBroadcast(messageIntent);

                        ByteBuffer buffer = ByteBuffer.allocate(4 * mmSensorValues.length);

                        for (float value : mmSensorValues) {
                            buffer.putFloat(value);
                        }

                        byte[] bytes = buffer.array();
                        Timestamp timestamp = new Timestamp(System.currentTimeMillis());

                        Event event = new Event(bytes, timestamp, UUID.fromString(getApplicationContext().getSharedPreferences("Fall_Detection", 0).getString("uuid", null)));
                        try {
                            startTime = System.nanoTime();

                            Prediction.makePrediction(event);

                            noOfEvents++;
                            endTime = System.nanoTime();
                            latency = TimeUnit.NANOSECONDS.toMillis(endTime - startTime);
                            total = total + latency;

                            predictions++;
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            }, 0, 32);

    }
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        start_time = System.currentTimeMillis();
        startTime1 = System.nanoTime();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel serviceChannel = new NotificationChannel(
                    "ForegroundServiceChannel",
                    "Foreground Service Channel",
                    NotificationManager.IMPORTANCE_DEFAULT
            );

            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(serviceChannel);
        }

        Intent notificationIntent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(this,
                0, notificationIntent, 0);

        Notification notification = new NotificationCompat.Builder(this, "ForegroundServiceChannel")
                .setContentTitle("Foreground Service")
                .setContentIntent(pendingIntent)
                .build();

        startForeground(1, notification);


        mmSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        // mmSensor = mmSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        accSensor = mmSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyroSensor = mmSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        // mmSensorManager.registerListener(this,mmSensor,10000);
        mmSensorManager.registerListener(this, accSensor, SensorManager.SENSOR_DELAY_FASTEST);
        mmSensorManager.registerListener(this, gyroSensor, SensorManager.SENSOR_DELAY_FASTEST);

        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (timer != null) {
            try {
                timer.cancel();     // stop any scheduled tasks
                timer.purge();      // remove any queued tasks
            } catch (Exception e) {
                Log.e(TAG, "Error stopping timer: " + e.getMessage());
            }
            timer = null; // clear reference for GC
        }

        if (mmSensorManager != null) {
            mmSensorManager.unregisterListener(this);
            mmSensorManager = null;
        }

        float end_time = System.currentTimeMillis();
        endTime1 = System.nanoTime();

        System.out.println(end_time-start_time + " A TIMEEEE " + predictions + " AVG " + (end_time-start_time)/predictions);
        System.out.println("NANO " + TimeUnit.NANOSECONDS.toMillis(endTime1 - startTime1)/predictions);
        System.out.println( "MY TOTAL :" + total+ "  ::"+total/predictions);
        System.out.println( "Total number of events: " + noOfEvents);
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        // mmSensorValues = event.values;
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            latestAcc = event.values.clone();
        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            latestGyro = event.values.clone();
        }

        // Fuse only when both sensors have data
        if (latestAcc != null && latestGyro != null) {
            if (mmSensorValues == null) mmSensorValues = new float[6];

            System.arraycopy(latestAcc, 0, mmSensorValues, 0, 3);
            System.arraycopy(latestGyro, 0, mmSensorValues, 3, 3);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }
}