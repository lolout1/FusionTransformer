package com.example.wear.Prediction;

import android.content.Context;
import android.util.Log;

import com.google.gson.Gson;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;

import com.example.wear.Database.Couchbase;
import com.example.wear.config.ModelConfig;

/**
 * This class is responsible for making inference if the personalization strategy is ENSEMBLE.
 *
 * @author Bhargav Balusu (b_b515)
 * @version 1.0
 * @since 2022.05.15
 */
public class PersonalizedPredictionEnsemble {

    private static final String TAG = "PersonalizedPredictionEnsemble";
    private static ArrayList<Interpreter> interpreters = new ArrayList<>();
    private static float[] MODEL_WEIGHTS = {1.0f};
    private static float[] MODEL_THRESHOLDS = {0.30f};



    /**
     * Initializes the interpreters with downloaded models present in the local storage.
     * @param context
     * @throws IOException
     */
    public static void initialize(Context context) throws IOException {
        checkAndDownloadNewModel(context);
        for(String fileName : ModelConfig.getModelConfig(context).modelNames){
            interpreters.add(new Interpreter(new File(context.getFilesDir().getAbsolutePath() + "/"+fileName)));
        }
        MODEL_THRESHOLDS = ModelConfig.getModelConfig(context).thresholds;

    }

    public static float getThreshold() {
        return MODEL_THRESHOLDS[0];
    }
    /**
     * This method is used to transforms samples from a two-dimensional float array
     * into a one-dimensional float array
     *
     * @param samples float: Two-dimensional float array
     * @return float: One-dimensional float array
     */
    public static float[] flattenInputs(float[][] samples) {

        float[] flattenedSamples = new float[samples.length * samples[0].length];
        for (int i = 0; i < flattenedSamples.length; i += samples[0].length) {
            for (int x = 0; x < samples[0].length; x++) {
                try {
                    flattenedSamples[i + x] = samples[ i / samples[0].length][x];
                }catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
            }
        }
        return flattenedSamples;
    }

    /**
     * This method performs a weighted sum inference on received float samples using
     * the list of interpreters.
     *
     * @param samples float: Two-dimensional float array of samples
     * @return float: The weighted sum inference made my the interpreters
     */
    public static float makeInference(float[][] samples) throws Exception {

        float inference = 0.0f;

        float[] flattenedSamples = flattenInputs(samples);
        float[][] outputs = new float[1][2];
        outputs[0][0] = 0f;
        outputs[0][1] = 0f;

        for (Interpreter interpreter : interpreters) {
            interpreter.run(flattenedSamples, outputs);   // make inference of beta samples
            inference += outputs[0][0] * MODEL_WEIGHTS[interpreters.indexOf(interpreter)];
        }

        return inference;
    }

    /**
     * This method calls the ModelDownloader and writes the model files to local storage.
     * @param context
     */

        private static void checkAndDownloadNewModel(Context context) {

        try {
            ModelConfig modelConfig = ModelConfig.getModelConfig(context);
            ModelConfig config = new ModelDownloader().execute(modelConfig).get();

            if(config.isDownloaded){
                String[] modelNames = config.modelNames;
                MODEL_WEIGHTS = config.modelWeights;
                byte[][] modelContent = config.modelContent;
                for(int i=0; i<modelNames.length; i++){
                    Log.e(TAG,modelContent[i].toString());


                    try (FileOutputStream fos = context.openFileOutput(modelNames[i], Context.MODE_PRIVATE)) {
                        fos.write(modelContent[i]);
                    }
                    catch (Exception e){
                        Log.e(TAG, "Error while writing model content : " + e.getMessage());
                    }
                    Log.e(TAG, context.getFilesDir().getAbsolutePath());
                }
                Couchbase.initialize(context);
                updateConfigFile(context);

            }
        }catch(Exception e){
            Log.e(TAG, "Exception while downloading new model : "+ e.getMessage());
        }

    }

    /**
     * This method serializes the ModelConfig object and writes the result to SmartWatchValues.json
     * @param context
     */
        public static void updateConfigFile(Context context){
        ModelConfig config = ModelConfig.getModelConfig(context);
        String filename =  "SmartWatchValues.json" ;

        Gson gson = new Gson();
        String jsonString = gson.toJson(config);

        try (FileOutputStream fos = context.openFileOutput(filename, Context.MODE_PRIVATE)) {
            fos.write(jsonString.getBytes());
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

}
