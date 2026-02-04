package com.example.wear;

import android.content.Intent;
import android.os.Bundle;

import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.wear.ambient.AmbientModeSupport;

public class HelpActivity extends AppCompatActivity implements AmbientModeSupport.AmbientCallbackProvider {


    private final static String TAG = "HelpActivity";
    private AmbientModeSupport.AmbientController ambientController;

    private TextView mTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_help);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        ambientController = AmbientModeSupport.attach(this);
        // Get the feedback buttons
        Button okBtn = (Button) findViewById(R.id.help_ok_btn);

        //Set the OnClick functions to send the corresponding feedback to the ble service
        okBtn.setOnClickListener(new View.OnClickListener() {
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
}