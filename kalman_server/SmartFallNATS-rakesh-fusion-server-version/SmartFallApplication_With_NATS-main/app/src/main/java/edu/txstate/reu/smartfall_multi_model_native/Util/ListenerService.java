package edu.txstate.reu.smartfall_multi_model_native.Util;

import android.content.Intent;
import android.util.Log;

import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import com.google.android.gms.wearable.MessageEvent;
import com.google.android.gms.wearable.WearableListenerService;


/*
 *  This is a listener on the  device to get messages via
 *  the datalayer and then pass it to the main activity so it can be
 *  displayed.  the messages should be coming from the wear/watch device.
 */

public class ListenerService extends WearableListenerService {
    String TAG = "mobile Listener";
    @Override
    public void onMessageReceived(MessageEvent messageEvent) {

        if (messageEvent.getPath().equals("/user/help")) {
            final String message = new String(messageEvent.getData());
            Log.v(TAG, "Message path received on phone is: " + messageEvent.getPath());
            Log.v(TAG, "Message received on phone is: " + message);

            // Broadcast message to MainActivity for display
            Intent messageIntent = new Intent("helpData");
            messageIntent.putExtra("message", message);
            LocalBroadcastManager.getInstance(this).sendBroadcast(messageIntent);
        }
        else {
            super.onMessageReceived(messageEvent);
        }
    }

}