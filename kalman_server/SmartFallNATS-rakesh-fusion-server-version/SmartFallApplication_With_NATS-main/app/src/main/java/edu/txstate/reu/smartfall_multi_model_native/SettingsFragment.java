package edu.txstate.reu.smartfall_multi_model_native;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentTransaction;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.android.gms.wearable.Node;
import com.google.android.gms.wearable.Wearable;
import com.google.gson.Gson;

import java.util.List;
import java.util.concurrent.ExecutionException;

public class SettingsFragment extends Fragment {
    String uuid;
    SharedPreferences pref;
    SharedPreferences.Editor editor;
    TextView textView;
    String datapath = "/user/uuid";
    String TAG = "Mobile MainActivity Settings";

    protected Handler handler;
    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        Window window = getActivity().getWindow();
        window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
        window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
        window.setStatusBarColor(ContextCompat.getColor(getActivity(),R.color.gray));

        //message handler for the send thread.
        handler = new Handler(new Handler.Callback() {
            @Override
            public boolean handleMessage(Message msg) {
                Bundle stuff = msg.getData();
                return true;
            }
        });

        View view  = inflater.inflate(R.layout.fragment_settings,container,false);
        textView = view.findViewById(R.id.uuid_tv);
        pref = container.getContext().getSharedPreferences("Fall_Detection",0);
        editor = pref.edit();
        uuid = pref.getString("uuid", null);
        String tracker = pref.getString("trackerId", null);

        if(uuid!=null){
            textView.setText(uuid);

//            new SendThread(datapath, uuid).start();
        }
        else {
            Toast.makeText(container.getContext(),"You need to setup an account first.",Toast.LENGTH_SHORT).show();

            FragmentTransaction fragmentTransaction = getActivity()
                    .getSupportFragmentManager().beginTransaction();
            fragmentTransaction.replace(R.id.fragment_container, new ProfileFragment());
            fragmentTransaction.commit();
        }

        return view;
    }
}
