package com.example.wear;

import android.app.Activity;
import android.content.Intent;
import android.media.MediaPlayer;
import android.media.Ringtone;
import android.media.RingtoneManager;
import android.net.Uri;
import android.nfc.Tag;
import android.os.Bundle;

import android.os.CountDownTimer;
import android.os.Vibrator;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;
import androidx.wear.ambient.AmbientModeSupport;

import com.example.wear.Database.Database;
import com.example.wear.Prediction.Prediction;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.android.gms.wearable.Node;
import com.google.android.gms.wearable.Wearable;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.List;
import java.util.concurrent.ExecutionException;

//import edu.txstate.reu.ble.BluetoothLe;

public class FeedbackActivity extends AppCompatActivity implements AmbientModeSupport.AmbientCallbackProvider {

    private static final long START_IN_MILLIS = 300000;          //300 seconds
    private final static String TAG = "Feedback Activity";
    private TextView mTextView;
    private CountDownTimer mcountDownTimer;                     //Actual timer
    private long timeLeftInMilliseconds = START_IN_MILLIS;
    private static Intent helpAlertIntent = null;
    private AmbientModeSupport.AmbientController ambientController;
    TextView countdownText;
    MediaPlayer player;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_feedback);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        ambientController = AmbientModeSupport.attach(this);
        Log.d(TAG, "onCreate: ");
        mTextView = (TextView) findViewById(R.id.text);

        // Get the feedback buttons
        Button yesBtn = (Button) findViewById(R.id.feedback_yes_btn);
        Button noBtn = (Button) findViewById(R.id.feedback_no_btn);
        countdownText = (TextView) findViewById(R.id.countdown);

        helpAlertIntent = new Intent(this, HelpAlertActivity.class);

        Vibrator vibrator = (Vibrator) getSystemService(VIBRATOR_SERVICE);
        long[] vibrationPattern = {0, 500, 50, 300};
        //-1 - don't repeat
        final int indexInPatternToRepeat = -1;
        vibrator.vibrate(vibrationPattern, indexInPatternToRepeat);

        try {
            Uri notification = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE);
            player = MediaPlayer.create(this, notification);
            player.setLooping(true);
            player.start();
        } catch (Exception e) {
            e.printStackTrace();
        }

        //Set the OnClick functions to send the corresponding feedback to the ble service
        yesBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                JSONObject updates = new JSONObject();
                try {
                    updates.put("type", "TP");
                    Database.updateDocument(updates);

                    Prediction.reset();

                    Intent messageIntent = new Intent("CollectionIntent");
                    LocalBroadcastManager.getInstance(getApplicationContext()).sendBroadcast(messageIntent);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                Intent intent1 = new Intent(FeedbackActivity.this, HelpAlertActivity.class);
                startActivity(intent1);

                finish();
            }
        });

        noBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mcountDownTimer.cancel();
                JSONObject updates = new JSONObject();
                try {
                    updates.put("type", "FP");
                    Database.updateDocument(updates);

                    Prediction.reset();
                    Intent messageIntent = new Intent("CollectionIntent");
                    LocalBroadcastManager.getInstance(getApplicationContext()).sendBroadcast(messageIntent);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                finish();
            }
        });

        startTimer();
    }

    @Override
    protected void onDestroy(){
        player.stop();
        super.onDestroy();
    }

    private void startTimer() {
        mcountDownTimer = new CountDownTimer(timeLeftInMilliseconds, 1000) {
            @Override
            public void onTick(long millisUntilFinished) {
                timeLeftInMilliseconds = millisUntilFinished;
                int seconds = (int) (timeLeftInMilliseconds / 1000) % 300;

                if (seconds == 1) {
                    JSONObject updates = new JSONObject();
                    try {
                        updates.put("type", "TP");
                        Database.updateDocument(updates);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    new SendThread(HelpAlertActivity.datapath, "help").start();
                    finish();
                }

                String timeLeftFormatted = String.format("%02d", seconds);
                countdownText.setText(timeLeftFormatted);
            }

            @Override
            public void onFinish() {

            }
        }.start();
    }

    @Override
    public AmbientModeSupport.AmbientCallback getAmbientCallback() {
        return new FeedbackActivity.MyAmbientCallback();
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

    class SendThread extends Thread {
        String path;
        String message;

        //constructor
        SendThread(String p, String msg) {
            path = p;
            message = msg;
        }

        //sends the message via the thread.  this will send to all wearables connected, but
        //since there is (should only?) be one, no problem.
        public void run() {

            //first get all the nodes, ie connected wearable devices.
            Task<List<Node>> nodeListTask =
                    Wearable.getNodeClient(getApplicationContext()).getConnectedNodes();
            try {
                // Block on a task and get the result synchronously (because this is on a background
                // thread).
                List<Node> nodes = Tasks.await(nodeListTask);

                //Now send the message to each device.
                for (Node node : nodes) {
                    Task<Integer> sendMessageTask =
                            Wearable.getMessageClient(FeedbackActivity.this).sendMessage(node.getId(), path, message.getBytes());

                    try {
                        // Block on a task and get the result synchronously (because this is on a background
                        // thread).
                        Integer result = Tasks.await(sendMessageTask);
                        Log.v(TAG, "SendThread: message send to " + node.getDisplayName());

                    } catch (ExecutionException exception) {
                        Log.e(TAG, "Send Task failed: " + exception);

                    } catch (InterruptedException exception) {
                        Log.e(TAG, "Send Interrupt occurred: " + exception);
                    }

                }

            } catch (ExecutionException exception) {
                Log.e(TAG, "Node Task failed: " + exception);

            } catch (InterruptedException exception) {
                Log.e(TAG, "Node Interrupt occurred: " + exception);
            }

        }
    }
}