package edu.txstate.reu.smartfall_multi_model_native;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.IntentSender;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.location.Location;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.widget.Toast;

import com.google.android.gms.common.api.ResolvableApiException;
import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.location.LocationSettingsRequest;
import com.google.android.gms.location.LocationSettingsResponse;
import com.google.android.gms.location.SettingsClient;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.google.android.material.navigation.NavigationBarView;
import edu.txstate.reu.smartfall_multi_model_native.Util.SendMessage;

public class MainActivity extends AppCompatActivity {

    public String name, recipient;
    private final String TAG = "MainActivity";

    private static final int REQUEST_CHECK_SETTINGS = 1;
    private static final int REQUEST_GRANT_PERMISSION = 2;
    private static FusedLocationProviderClient fusedLocationClient;
    LocationRequest locationRequest;
    private Location currentLocation;
    private LocationCallback locationCallback;

    private SendMessage email = new SendMessage();
    public static boolean predictionInitialized = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this);
        setContentView(R.layout.activity_main);

        IntentFilter messageFilter = new IntentFilter("HelpData");
        MessageReceiver messageReceiver = new MessageReceiver();
        LocalBroadcastManager.getInstance(this).registerReceiver(messageReceiver, messageFilter);

        /** Bottom Navbar **/
        getSupportFragmentManager().beginTransaction().replace(R.id.fragment_container, new HomeFragment()).commit();
        BottomNavigationView bottomNavigationView = findViewById(R.id.bottomNavigationView);
        bottomNavigationView.setOnItemSelectedListener(new NavigationBarView.OnItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(@NonNull MenuItem item) {
                Fragment fragment = null;

                switch (item.getItemId()) {
                    case R.id.nav_home:
                        fragment = new HomeFragment();
                        break;
                    case R.id.nav_profile:
                        fragment = new ProfileFragment();
                        break;
                    case R.id.nav_settings:
                        fragment = new SettingsFragment();
                        break;
                }

                getSupportFragmentManager().beginTransaction().replace(R.id.fragment_container,
                        fragment).commit();

                return true;
            }
        });

    }

    public class MessageReceiver extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            sendEmergencyEmail();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    private BroadcastReceiver helpDataReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            sendEmergencyEmail();
        }
    };

    public void sendEmergencyEmail() {
        SharedPreferences pref = getApplicationContext().getSharedPreferences("Fall_Detection",0);
        SharedPreferences.Editor editor = pref.edit();
        name = pref.getString("p_name", null);
        recipient = pref.getString("e_email", null);
        createLocationRequest();
        settingsCheck();
    }

    protected void createLocationRequest() {
        locationRequest = LocationRequest.create();
        locationRequest.setInterval(10000);
        locationRequest.setFastestInterval(5000);
        locationRequest.setPriority(LocationRequest.PRIORITY_HIGH_ACCURACY);
    }

    public void settingsCheck() {
        LocationSettingsRequest.Builder builder = new LocationSettingsRequest.Builder()
                .addLocationRequest(locationRequest);

        SettingsClient client = LocationServices.getSettingsClient(this);
        Task<LocationSettingsResponse> task = client.checkLocationSettings(builder.build());
        ((Task) task).addOnSuccessListener(this, new OnSuccessListener<LocationSettingsResponse>() {
            @Override
            public void onSuccess(LocationSettingsResponse locationSettingsResponse) {
                // All location settings are satisfied. The client can initialize
                // location requests here.
                Log.d("TAG", "onSuccess: settingsCheck");
                getCurrentLocation();
            }
        });

        task.addOnFailureListener(this, new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception e) {
                if (e instanceof ResolvableApiException) {
                    // Location settings are not satisfied, but this can be fixed
                    // by showing the user a dialog.
                    Log.d("TAG", "onFailure: settingsCheck");
                    try {
                        // Show the dialog by calling startResolutionForResult(),
                        // and check the result in onActivityResult().
                        ResolvableApiException resolvable = (ResolvableApiException) e;
                        resolvable.startResolutionForResult(MainActivity.this,
                                REQUEST_CHECK_SETTINGS);
                    } catch (IntentSender.SendIntentException sendEx) {
                        // Ignore the error.
                    }
                }
            }
        });
    }

    public void getCurrentLocation(){
        if (this.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && this.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            this.requestPermissions(new String[]{Manifest.permission.ACCESS_FINE_LOCATION},1);
            return;
        }
        fusedLocationClient.getLastLocation()
                .addOnSuccessListener(this, new OnSuccessListener<Location>() {
                    @Override
                    public void onSuccess(Location location) {
                        Log.d("TAG", "onSuccess: getLastLocation");
                        // Got last known location. In some rare situations this can be null.
                        if (location != null) {
                            currentLocation=location;
                            Log.d("TAG", "onSuccess:latitude "+location.getLatitude());
                            Log.d("TAG", "onSuccess:longitude "+location.getLongitude());
                            try {
                                email.sendEmail(name, recipient,currentLocation.getLatitude(),currentLocation.getLongitude());
                            }
                            catch (Exception e) {
                                Log.d("Exception", "Email sending exception occured.");
                            }
                        }else{
                            Log.d("TAG", "location is null");
                            buildLocationCallback();
                        }
                    }
                });
    }

    private void buildLocationCallback() {
        locationCallback = new LocationCallback() {
            @Override
            public void onLocationResult(LocationResult locationResult) {
                if (locationResult == null) {
                    Log.d("TAG", "location is null");
                    return;
                }
                for (android.location.Location location : locationResult.getLocations()) {
                    /*
                     *  Update UI with location data
                     */
                    currentLocation=location;
                    Log.d("TAG", "Latitude: "+currentLocation.getLatitude());
                    Log.d("TAG", "Longitude: "+currentLocation.getLongitude());
                    try {
                        email.sendEmail(name, recipient,currentLocation.getLatitude(),currentLocation.getLongitude());
                    }
                    catch (Exception e) {
                        Log.d("Exception", "Email sending exception occured.");
                    }
                }
                if(locationCallback!=null)
                    fusedLocationClient.removeLocationUpdates(locationCallback);
            };
        };
    }

    /*
     * called after user responds to location permission popup
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(requestCode==REQUEST_GRANT_PERMISSION){
            getCurrentLocation();
        }
    }
    /*
     * called after user responds to location settings popup
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Log.d("TAG", "onActivityResult: ");
        if(requestCode==REQUEST_CHECK_SETTINGS && resultCode==RESULT_OK)
            getCurrentLocation();
        if(requestCode==REQUEST_CHECK_SETTINGS && resultCode==RESULT_CANCELED)
            Toast.makeText(this, "Please enable Location settings...!!!", Toast.LENGTH_SHORT).show();
    }
}