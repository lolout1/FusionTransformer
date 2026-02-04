package com.example.wear;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;

import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.wear.ambient.AmbientModeSupport;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.android.gms.wearable.Node;
import com.google.android.gms.wearable.Wearable;

import java.util.List;
import java.util.concurrent.ExecutionException;

public class HelpAlertActivity extends AppCompatActivity implements AmbientModeSupport.AmbientCallbackProvider {
    public static String datapath = "/user/uuid";
    String DATA_TAG = "Mobile MainActivity";
    protected Handler handler;

    private final static String TAG = "HelpAlertActivity";
    private AmbientModeSupport.AmbientController ambientController;
    private static Intent helpintent = null;
    private TextView mTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_alert);

        mTextView = (TextView) findViewById(R.id.text);
        helpintent = new Intent(this, HelpActivity.class);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        ambientController = AmbientModeSupport.attach(this);
        // Get the feedback buttons
        Button yesBtn = (Button) findViewById(R.id.help_yes_btn);
        Button noBtn = (Button) findViewById(R.id.help_no_btn);

        //message handler for the send thread.
        handler = new Handler(new Handler.Callback() {
            @Override
            public boolean handleMessage(Message msg) {
                Bundle stuff = msg.getData();
                return true;
            }
        });

        //Set the OnClick functions to send the corresponding feedback to the ble service
        yesBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                new SendThread(datapath, "help").start();
                startActivity(helpintent);
                finish();
            }
        });

        noBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
            }
        });

    }

    @Override
    public AmbientModeSupport.AmbientCallback getAmbientCallback() {
        return new HelpActivity.MyAmbientCallback();
    }

    public static class MyAmbientCallback extends AmbientModeSupport.AmbientCallback {
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

    //This actually sends the message to the wearable device.
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
                            Wearable.getMessageClient(HelpAlertActivity.this).sendMessage(node.getId(), path, message.getBytes());

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