package com.example.wear;

import android.Manifest;
import android.app.Activity;

import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.ServiceConnection;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.NetworkInfo;
import android.net.NetworkRequest;
import android.net.wifi.WifiManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.text.Layout;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;
import androidx.wear.ambient.AmbientMode;
import androidx.wear.ambient.AmbientModeSupport;

import com.example.wear.Database.Database;
import com.example.wear.Prediction.Prediction;
import com.example.wear.config.ModelConfig;
import com.example.wear.config.SmartFallConfig;
import com.example.wear.util.Event;
import com.google.android.gms.wearable.MessageClient;
import com.google.android.gms.wearable.WearableListenerService;
import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton;
import com.google.android.material.floatingactionbutton.FloatingActionButton;

import java.nio.ByteBuffer;
import java.sql.Timestamp;
import java.util.UUID;
import io.nats.client.Connection;
import android.net.ConnectivityManager;
import android.content.pm.PackageManager;
import android.os.Build;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;

public class MainActivity extends AppCompatActivity implements //BleServer.OnEventReceivedListener,
        AmbientModeSupport.AmbientCallbackProvider {
    private ConnectivityManager connectivityManager;
    private ConnectivityManager.NetworkCallback networkCallback;

    /**
     * A string TAG for debugging
     */
    private NatsManager nats;
    private final String TAG = "MainActivity";

    public static boolean predictionInitialized = false;

    public static String uuid = null;

    public static boolean feedbackEnabled = true;

    /**
     *
     */
    private AmbientModeSupport.AmbientController ambientController;

    /**
     *
     */
    Intent sensorIntent;


    /**
     * A constant integer representing that the received feedback label is a false positive label
     */
    public static final int FEEDBACK_LABEL_FP = 0;

    /**
     * A constant integer representing that the received feedback is a true positive label
     */
    public static final int FEEDBACK_LABEL_TP = 1;
    /**
     * A constant integer representing that the received feedback is a false negative label
     */
    private static final int FEEDBACK_LABEL_FN = 3;

    /**
     * A constant integer representing that the received feedback is a request for help
     */
    public static final int FEEDBACK_NEEDS_HELP = 2;

    /**
     * A constant integer representing that the state of the data collection is actively collecting
     */
    public static final int COLLECTION_STATE = 0;

    /**
     * A constant integer representing that the state of the data collection is awaiting feedback
     */
    public static final int FEEDBACK_STATE = 1;

    /**
     * A constant integer representing that the state of the data collection is inactive
     */
    public static final int IDLE_STATE = 2;

    /**
     * A constant string representing the designated path for receiving accelerometer data
     */
    public static final String ACCELEROMETER_EVENT_PATH = "accelerometer";

    /**
     * A constant string representing the designated path for receiving state data
     */
    public static final String STATE_EVENT_PATH = "state";

    /**
     * A constant string representing the designated path for receiving feedback data
     */
    public static final String FEEDBACK_EVENT_PATH = "feedback";

    public TextView tv2;
    private static final int PERMISSION_REQUEST_CODE = 1;
    public static String watchStatus;

    /**
     * A list of all declared paths used for sending and receiving ble data
     */
    private static String [] blePaths = new String []
            { ACCELEROMETER_EVENT_PATH, STATE_EVENT_PATH, FEEDBACK_EVENT_PATH };

    public Context context;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        context = getApplicationContext();
        initializeAll(getApplicationContext());

       /* ConnectivityManager cm = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo activeNetwork = cm.getActiveNetworkInfo();
        if (activeNetwork != null && activeNetwork.isConnected()) {
            if (activeNetwork.getType() == ConnectivityManager.TYPE_WIFI) {
                nats.connect();
            } else if (activeNetwork.getType() == ConnectivityManager.TYPE_BLUETOOTH) {
                // Handle Bluetooth connection logic if necessary
                Log.d(TAG, "Connected via Bluetooth.");
            }
        } else {
            Log.d(TAG, "No active connection.");
        }*/

        setContentView(R.layout.activity_main);
        //nats = new NatsManager(this);
        nats = new NatsManager(this,context);
        System.out.println("Calling connect");
        //nats.connect(this);
        nats.connect();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        ambientController = AmbientModeSupport.attach(this);

        SharedPreferences pref;
        SharedPreferences.Editor editor;
        pref = getApplicationContext().getSharedPreferences("Fall_Detection",0);
        editor = pref.edit();
        String message = pref.getString("uuid",null);
        if (message == null) {
            System.out.println("onCreate - uuid is null, creating new uuid");
            message = UUID.randomUUID().toString();
        }

        System.out.println("onCreate - Setting uuid: " + message);
        // message = "2dc69f4b-440f-4395-8898-6a05cc4ec313";
        // System.out.println("After setting uuid: " + message);

        editor.putString("uuid", message);
        editor.commit();

        ExtendedFloatingActionButton assitBtn = (ExtendedFloatingActionButton) findViewById(R.id.assitBtn);
        assitBtn.setText("FELL");
        assitBtn.setBackgroundColor(Color.RED);
        assitBtn.setTextAlignment(View.TEXT_ALIGNMENT_CENTER);

        assitBtn.setOnClickListener(v -> {
            Log.w(TAG, "FELL button clicked on watch");
        
            try {
                Prediction.onFellClickedInWatch();
                Toast.makeText(this, "Fall marked manually", Toast.LENGTH_SHORT).show();
            } catch (Exception e) {
                Log.e(TAG, "Error in manual fall click", e);
            }
        });
        

        tv2 = findViewById(R.id.textView2);
        tv2.setVisibility(TextView.INVISIBLE);
        tv2.setText("Deactivated");
//        assitBtn.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                assitBtn.setText("FELL");
//                assitBtn.setBackgroundColor(Color.RED);
//            }
//        });

        IntentFilter messageFilter = new IntentFilter(Intent.ACTION_SEND);
        MessageReceiver messageReceiver = new MessageReceiver();
        LocalBroadcastManager.getInstance(this).registerReceiver(messageReceiver, messageFilter);

        IntentFilter dataReceiverFilter = new IntentFilter("data-receiver");
        DataReceiver dataReceiver = new DataReceiver();
        LocalBroadcastManager.getInstance(this).registerReceiver(dataReceiver, dataReceiverFilter);

        IntentFilter sensorDataFilter = new IntentFilter("sensor-data");
        SensorDataReceiver sensorDataReceiver = new SensorDataReceiver();
        LocalBroadcastManager.getInstance(this).registerReceiver(sensorDataReceiver, sensorDataFilter);

        IntentFilter feedbackFilter = new IntentFilter("FeedbackIntent");
        FeedbackReceiver feedbackReceiver = new FeedbackReceiver();
        LocalBroadcastManager.getInstance(this).registerReceiver(feedbackReceiver, feedbackFilter);

        IntentFilter collectionFilter = new IntentFilter("CollectionIntent");
        CollectionReceiver collectionReceiver = new CollectionReceiver();
        LocalBroadcastManager.getInstance(this).registerReceiver(collectionReceiver, collectionFilter);


        IntentFilter IdleFilter = new IntentFilter("IdleIntent");
        IdleReceiver IdleReceiver = new IdleReceiver();
        LocalBroadcastManager.getInstance(this).registerReceiver(IdleReceiver, IdleFilter);

    }

    public static void  initializeAll(Context context){
        try {
            /** Database **/
            Database.initialize(context);
            /** Prediction **/
            Prediction.initialize(context);
            predictionInitialized = true;
            ModelConfig config = ModelConfig.getModelConfig(context);
            config.uuid = context.getSharedPreferences("Fall_Detection",0).getString("uuid",null);

            System.out.println("initializeAll - Setting uuid: " + config.uuid);
            // config.uuid = "2dc69f4b-440f-4395-8898-6a05cc4ec313";
            // System.out.println("After setting uuid: " + config.uuid);

            if(!SmartFallConfig.MODEL_TYPE.equals("PERSONALIZED")){
                config.modelVersion = 0;
                config.newDocID = SmartFallConfig.MODEL_TYPE.toLowerCase();
                config.modelContent = null;
                config.modelWeights = null;
            }
            Prediction.updateTracker(context);
        } catch (Exception e){
            e.printStackTrace();
        }

    }

    @Override
    protected void onDestroy() {
        Log.d(TAG, "onDestroy: ");
        super.onDestroy();
    }

    @Override
    public AmbientModeSupport.AmbientCallback getAmbientCallback() {
        return new MyAmbientCallback();
    }

    private class MyAmbientCallback extends AmbientModeSupport.AmbientCallback {
        @Override
        public void onEnterAmbient(Bundle ambientDetails) {
            // Handle entering ambient mode
        }

        @Override
        public void onExitAmbient() {
            // Handle exiting ambient mode
        }

        @Override
        public void onUpdateAmbient() {
            // Update the content
        }
    }


    // Register the local broadcast receiver to receive messages from the listener.
    public class MessageReceiver extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {


            String path = intent.getStringExtra("path");
            String message = intent.getStringExtra("message");
            Log.v(TAG, "Main activity received message: " + message);
            Log.v(TAG, "Main activity received path: " + path);

            if(path.equals("/user/uuid")) {

                SharedPreferences pref;
                SharedPreferences.Editor editor;
                pref = getApplicationContext().getSharedPreferences("Fall_Detection", 0);
                editor = pref.edit();

                System.out.println("onReceive - Setting uuid: " + message);
                // message = "2dc69f4b-440f-4395-8898-6a05cc4ec313";
                // System.out.println("After - setting uuid: " + message);

                editor.putString("uuid", message);
                editor.commit();
            }
            else {
                if(message.equals("on")) {
                    feedbackEnabled = true;
                    initializeAll(getApplicationContext());
                    sensorIntent = new Intent(MainActivity.this, SensorService.class );
                    ContextCompat.startForegroundService(MainActivity.this, sensorIntent);
                    tv2.setText("Activated");
                    watchStatus = tv2.getText().toString();
                    System.out.println("activated the watch");
                    tv2.setVisibility(TextView.VISIBLE);
                }
                else {
                    feedbackEnabled = false;
                    Prediction.uploadTracker(context);
                    stopService(sensorIntent);
                    Prediction.reset();
                    tv2.setText("Deactivated");
                    watchStatus = tv2.getText().toString();
                    System.out.println("Deactivated the watch");
                    tv2.setVisibility(TextView.INVISIBLE);
                }
            }
        }
    }

    public class DataReceiver extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            String message = intent.getStringExtra("message");
//            Log.v(TAG, "Main activity received message: " + message);

            tv2.setText(message);
        }
    }

    public class FeedbackReceiver extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            if(feedbackEnabled) {

                feedbackEnabled = false;
                //System.out.println("ittt");
                String message = intent.getStringExtra("message");
                Log.v(TAG, "Main activity received message: " + message);
                stopService(sensorIntent);
                Prediction.reset();
//            initializeAll(context);
                Intent intent1 = new Intent(MainActivity.this, FeedbackActivity.class);
                startActivity(intent1);
                //System.out.println("isss");
            }
        }
    }

    public class CollectionReceiver extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            feedbackEnabled = true;
            String message = intent.getStringExtra("message");
            Log.v(TAG, "Main activity received message: " + message);
            Prediction.reset();
            initializeAll(context);
            sensorIntent = new Intent(MainActivity.this, SensorService.class );
            ContextCompat.startForegroundService(MainActivity.this, sensorIntent);
        }
    }

    public class IdleReceiver extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            String message = intent.getStringExtra("message");
            Log.v(TAG, "Main activity received message: " + message);
            feedbackEnabled = false;
            stopService(sensorIntent);
        }
    }

    public class SensorDataReceiver extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            String data = intent.getStringExtra("data");

            String[] parts = data.split(",");
            float[] mmSensorValues = new float[parts.length];
            for (int i = 0; i < parts.length; i++) {
                try {
                    mmSensorValues[i] = Float.parseFloat(parts[i]);
                } catch (NumberFormatException e) {
                    // Handle the exception (e.g., print an error message or throw the exception)
                    e.printStackTrace();
                }
            }
            ByteBuffer buffer = ByteBuffer.allocate(4 * mmSensorValues.length);

            for (float value : mmSensorValues) {
                buffer.putFloat(value);
            }

            byte[] bytes = buffer.array();
            Timestamp timestamp = new Timestamp(System.currentTimeMillis());

            Event event = new Event(bytes, timestamp, UUID.fromString(getApplicationContext().getSharedPreferences("Fall_Detection", 0).getString("uuid", null)));

            try {
                Prediction.makePrediction(event);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}