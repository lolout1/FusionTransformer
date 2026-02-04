package edu.txstate.reu.smartfall_multi_model_native;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentTransaction;

import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton;
import com.google.android.material.textfield.TextInputLayout;

import java.util.UUID;

public class ProfileFragment extends Fragment {
    private TextInputLayout name, email, phone, contact_name, contact_email, contact_phone;
    public static final String EXTRA_UUID = "reuiot.smartwatch.ProfileSettingsActivity.EXTRA_UUID";
    String uuid;
    SharedPreferences pref;
    SharedPreferences.Editor editor;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        // Link up the views.
        Window window = getActivity().getWindow();
        window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
        window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
        window.setStatusBarColor(ContextCompat.getColor(getActivity(),R.color.gray));

        View view  = inflater.inflate(R.layout.fragment_profile,container,false);
        ExtendedFloatingActionButton save_button = (ExtendedFloatingActionButton) view.findViewById(R.id.save_changes);
        name = (TextInputLayout) view.findViewById(R.id.p_name);
        email = (TextInputLayout) view.findViewById(R.id.p_email);
        phone = (TextInputLayout) view.findViewById(R.id.p_phone);
        contact_name = (TextInputLayout) view.findViewById(R.id.e_name);
        contact_email = (TextInputLayout) view.findViewById(R.id.e_email);
        contact_phone = (TextInputLayout) view.findViewById(R.id.e_phone);

        pref = container.getContext().getSharedPreferences("Fall_Detection",0);
        editor = pref.edit();

        String p_name = pref.getString("p_name", null);
        String p_email = pref.getString("p_email", null);
        String p_phone = pref.getString("p_phone", null);
        String e_name = pref.getString("e_name", null);
        String e_email = pref.getString("e_email", null);
        String e_phone = pref.getString("e_phone", null);
        uuid = pref.getString("uuid", null);

        if(uuid!=null){
            name.getEditText().setText(p_name);
            email.getEditText().setText(p_email);
            phone.getEditText().setText(p_phone);
            contact_name.getEditText().setText(e_name);
            contact_email.getEditText().setText(e_email);
            contact_phone.getEditText().setText(e_phone);
        }
        else {
            uuid = UUID.randomUUID().toString();

//            FragmentTransaction fragmentTransaction = getActivity()
//                    .getSupportFragmentManager().beginTransaction();
//            fragmentTransaction.replace(R.id.fragment_container, new HomeFragment());
//            fragmentTransaction.commit();
        }


        save_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                editor.putString("p_name", name.getEditText().getText().toString());
                editor.putString("p_email", email.getEditText().getText().toString());
                editor.putString("p_phone", phone.getEditText().getText().toString());
                editor.putString("e_name", contact_name.getEditText().getText().toString());
                editor.putString("e_email", contact_email.getEditText().getText().toString());
                editor.putString("e_phone", contact_phone.getEditText().getText().toString());
                editor.putString("uuid", uuid);
                editor.commit();

                Intent data = new Intent();
                data.putExtra(EXTRA_UUID, uuid);

                Toast.makeText(container.getContext(),"Information saved.",Toast.LENGTH_SHORT).show();

                FragmentTransaction fragmentTransaction = getActivity()
                        .getSupportFragmentManager().beginTransaction();
                fragmentTransaction.replace(R.id.fragment_container, new HomeFragment());
                fragmentTransaction.commit();
            }
        });

        return view;
    }
}
