package edu.txstate.reu.smartfall_multi_model_native;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.util.TypedValue;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentTransaction;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.android.gms.wearable.Node;
import com.google.android.gms.wearable.Wearable;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton;
import com.google.gson.Gson;

import java.util.List;
import java.util.concurrent.ExecutionException;


public class HomeFragment extends Fragment {
    public static final String EXTRA_UUID = "reuiot.smartwatch.ProfileSettingsActivity.EXTRA_UUID";

    public static long start_time;
    SharedPreferences pref;
    SharedPreferences.Editor editor;
    public String power;
    public TextView accDataView;

    String TAG = "Mobile MainActivity Home";

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        Window window = getActivity().getWindow();
        window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
        window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
        window.setStatusBarColor(ContextCompat.getColor(getActivity(), R.color.red));

        View view  = inflater.inflate(R.layout.fragment_home,container,false);
        MaterialCardView materialCardView = view.findViewById(R.id.fall_detection_mcv);
        TextView textView = view.findViewById(R.id.status_tv);
        accDataView = view.findViewById(R.id.acc_data);
        ExtendedFloatingActionButton powerFab = (ExtendedFloatingActionButton) view.findViewById(R.id.power_fab);

        pref = container.getContext().getSharedPreferences("Fall_Detection",0);
        editor = pref.edit();
        String uuid = pref.getString("uuid", null);
        if(uuid!=null){
            Intent data = new Intent();
            data.putExtra(EXTRA_UUID, uuid);

            new SendThread("/user/uuid", uuid).start();
        }
        else{
            Toast.makeText(container.getContext(),"You need to setup an account first.",Toast.LENGTH_SHORT).show();

            FragmentTransaction fragmentTransaction = getActivity()
                    .getSupportFragmentManager().beginTransaction();
            fragmentTransaction.replace(R.id.fragment_container, new ProfileFragment());
            fragmentTransaction.commit();
        }
        power = pref.getString("power", null);

        if(power==null || power.equals("off")){

            window.setStatusBarColor(ContextCompat.getColor(getActivity(), R.color.red));

            accDataView.setVisibility(View.INVISIBLE);
            powerFab.setText("Activate");
            powerFab.setTextSize(TypedValue.COMPLEX_UNIT_SP, 32);
            powerFab.setBackgroundTintList(ColorStateList.valueOf(Color
                    .parseColor("#29C58C")));

            materialCardView.setCardBackgroundColor(Color
                    .parseColor("#E20826"));
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                materialCardView.setOutlineAmbientShadowColor(Color
                        .parseColor("#E20826"));
                materialCardView.setOutlineSpotShadowColor(Color
                        .parseColor("#E20826"));
            }
            textView.setText("Off");
        }
        else {
            window.setStatusBarColor(ContextCompat.getColor(getActivity(), R.color.green));

            powerFab.setText("Deactivate");
            // new SendThread("/user/uuid", uuid).start();

            accDataView.setVisibility(View.VISIBLE);
            powerFab.setTextSize(TypedValue.COMPLEX_UNIT_SP, 26);

            powerFab.setBackgroundTintList(ColorStateList.valueOf(Color
                    .parseColor("#E20826")));

            materialCardView.setCardBackgroundColor(Color
                    .parseColor("#29C58C"));
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                materialCardView.setOutlineAmbientShadowColor(Color
                        .parseColor("#29C58C"));
                materialCardView.setOutlineSpotShadowColor(Color
                        .parseColor("#29C58C"));
            }
            textView.setText("On");
        }
        powerFab.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.R)
            @Override
            public void onClick(View v) {
                if(power==null || power.equals("off")){
                    window.setStatusBarColor(ContextCompat.getColor(getActivity(), R.color.green));
//                    accDataView.setVisibility(View.VISIBLE);
                    powerFab.setText("Deactivate");
                    // new SendThread("/user/uuid", uuid).start();
                    powerFab.setTextSize(TypedValue.COMPLEX_UNIT_SP, 26);

                    powerFab.setBackgroundTintList(ColorStateList.valueOf(Color
                            .parseColor("#E20826")));

                    materialCardView.setCardBackgroundColor(Color
                            .parseColor("#29C58C"));
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                        materialCardView.setOutlineAmbientShadowColor(Color
                                .parseColor("#29C58C"));
                        materialCardView.setOutlineSpotShadowColor(Color
                                .parseColor("#29C58C"));
                    }
                    textView.setText("On");

                    power = "on";

                    editor.putString("power", "on");
                    editor.commit();

                    new SendThread("/user/power", "on").start();

                    String uuid = pref.getString("uuid", null);
                    if(uuid != null) {
                        new SendThread("/user/uuid", uuid).start();
                    }
                }
                else {

                    accDataView.setVisibility(View.INVISIBLE);
                    window.setStatusBarColor(ContextCompat.getColor(getActivity(), R.color.red));

                    powerFab.setText("Activate");
                    powerFab.setTextSize(TypedValue.COMPLEX_UNIT_SP, 32);

                    powerFab.setBackgroundTintList(ColorStateList.valueOf(Color
                            .parseColor("#29C58C")));

                    materialCardView.setCardBackgroundColor(Color
                            .parseColor("#E20826"));
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                        materialCardView.setOutlineAmbientShadowColor(Color
                                .parseColor("#E20826"));
                        materialCardView.setOutlineSpotShadowColor(Color
                                .parseColor("#E20826"));
                    }
                    textView.setText("Off");

                    power = "off";

                    editor.putString("power", "off");
                    editor.commit();

                    new SendThread("/user/power", "off").start();
                }
            }
        });
        return view;
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
                    Wearable.getNodeClient(getActivity().getApplicationContext()).getConnectedNodes();
            try {
                // Block on a task and get the result synchronously (because this is on a background
                // thread).
                List<Node> nodes = Tasks.await(nodeListTask);

                //Now send the message to each device.
                for (Node node : nodes) {
                    Task<Integer> sendMessageTask =
                            Wearable.getMessageClient(getActivity()).sendMessage(node.getId(), path, message.getBytes());

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
