package com.example.wear.config;

import android.content.Context;
import android.content.Intent;
import android.util.Log;

import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import com.google.gson.Gson;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * This class holds all the details about the current model being used for prediction. This is a singleton class
 * and hence only single object is created and same reference is returned whenever the object is requested. This object is
 * updated whenever the strategy is changed or new model is downloaded. The object is serialized and stored to local file system for
 * persistence of the model details.
 *
 * @author Bhargav Balusu (b_b515)
 * @version 1.0
 * @since 2022.05.15
 */

public class ModelConfig {

    private static final String TAG = "ModelConfig";

    public String[] modelNames;
    public float[] modelWeights;
    public boolean isDownloaded;
    public byte[][] modelContent;
    public String oldDocID;
    public String newDocID;
    public String firstDocId;
    public String lastDocId;
    public String uuid;
    public int modelVersion;
    public String trackerId;
    public float[] thresholds;

    /**
     * Private constructor to make the class singleton.
     */
    private ModelConfig(){

    }
    private static ModelConfig modelConfig = null;

    /**
     * This is a static method that returns object refernce. If the object is not initialized, then the file SmartWatchValues.json is deserialized to
     * ModelConfig object. Else, it just returns the object refernce.
     * @param context
     * @return
     */
    public static ModelConfig getModelConfig(Context context){
        if(modelConfig == null){
            loadConfigFile(context);
        }

        return modelConfig;
    }

    public static ModelConfig getModelConfig(){
        if(modelConfig == null){
            modelConfig = new ModelConfig();
        }

        return modelConfig;
    }

    /**\
     * This method deserializes SmartWatchValues.json file to ModelConfig object. If the file is not present in the local storage, then default object
     * is returned back.
     * @param context
     */
    private static void loadConfigFile(Context context){
        FileInputStream fis = null;
        try {
            fis = context.openFileInput("SmartWatchValues.json");
        } catch (FileNotFoundException e) {
            Log.e(TAG,"Config file not found, returning default config object");
            modelConfig = new ModelConfig();
            return;
        }
        InputStreamReader inputStreamReader =
                new InputStreamReader(fis, StandardCharsets.UTF_8);
        StringBuilder stringBuilder = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(inputStreamReader)) {
            String line = reader.readLine();
            while (line != null) {
                stringBuilder.append(line).append('\n');
                line = reader.readLine();
            }

            String fileContents = stringBuilder.toString();

            modelConfig = new Gson().fromJson(fileContents,ModelConfig.class);

            Intent modelIntent = new Intent("modelInfo");
            modelIntent.putExtra("data", fileContents);
            LocalBroadcastManager.getInstance(context).sendBroadcast(modelIntent);
        } catch (IOException e) {
            Log.e(TAG,"Error while reading config file : " + e.getMessage());
            modelConfig = new ModelConfig();
            return;
        }



    }



}