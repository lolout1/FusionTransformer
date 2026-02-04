package com.example.wear.Prediction;

import android.os.AsyncTask;
import android.util.Log;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import com.example.wear.Database.Couchbase;
import com.example.wear.config.ModelConfig;
import com.example.wear.config.SmartFallConfig;

/**
 * This class is responsible for downloading the new models and updating the ModelConfig object.
 * The returned ModelConfig object will have all the details of the model that system should use to predict.
 *
 * @author Bhargav Balusu (b_b515)
 * @version 1.0
 * @since 2022.05.15
 */
public class ModelDownloader extends AsyncTask<ModelConfig , Integer, ModelConfig> {

    private static final String TAG = "ModelDownloader";
    @Override
    protected ModelConfig doInBackground(ModelConfig... urls) {

        SmartFallConfig.CloudIPConfig downloadConfig = SmartFallConfig.getCloudConfig();
        ModelConfig config = urls[0];
        config.isDownloaded = false;
        String oldDocID = config.newDocID;
        InputStream in = null;
        try {
            URL url = new URL(downloadConfig.checkVersion);/** call to checkversion.php script**/
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setReadTimeout(15000);         //15 seconds
            conn.setConnectTimeout(15000);      //15 seconds
            conn.addRequestProperty("Authorization", SmartFallConfig.API_KEY);
            conn.setDoInput(true);
            conn.setDoOutput(true);
            /**
             * For first install, default model is downloaded based on personalization strategy.
             */
            String reqData = "uuid"  + "=" + (SmartFallConfig.isFirst ? (SmartFallConfig.PERSONALIZED_STRATEGY.equals("LSTM") ? "default-lstm-2d" : "default-ensemble") : config.uuid);
            conn.getOutputStream().write(reqData.getBytes());
            int responseCode = conn.getResponseCode();

            if (responseCode == HttpURLConnection.HTTP_OK) {
                Log.d(TAG, "Successful checkversion call");
                BufferedReader connection_reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                StringBuffer sb = new StringBuffer();
                String line = " ";

                while ((line = connection_reader.readLine()) != null) {
//                    line = line.replaceAll("/data/", "_data_");
                    Log.d("print ", line);
                    sb.append(line);
                    break;
                }
                Log.d(TAG, "sb "+sb.toString());
                config.newDocID = sb.toString().replaceAll("'", "");
                if(config.newDocID.equals(""))
                    config.newDocID = SmartFallConfig.PERSONALIZED_STRATEGY.equals("LSTM") ? "default-lstm-2d" : "default-ensemble";
                connection_reader.close();
            } else {

                Log.e(TAG, "Connection response code was: NOT OKAY! models were not checked!");
            }

            if(!config.newDocID.equals(oldDocID)){
                Log.d("Model download", "Called");
                url = new URL(downloadConfig.downloadModel);//call to downloadModels.php
                conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setReadTimeout(15000);         //15 seconds
                conn.setConnectTimeout(15000);      //15 seconds
                conn.addRequestProperty("Authorization", SmartFallConfig.API_KEY);
                conn.setDoInput(true);
                conn.setDoOutput(true);
                String tempDocID = config.newDocID;
                reqData = "docid" + "=" + tempDocID;
                conn.getOutputStream().write(reqData.getBytes());
                responseCode = conn.getResponseCode();
                String contents = "";
                List<String[]> models = new ArrayList<String[]>();
                if(responseCode == HttpURLConnection.HTTP_OK) {
                    Log.d(TAG, "Successful downloadModels call");
                    Log.d("Model download", "Called");
                    BufferedReader connection_reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                    StringBuffer sb = new StringBuffer();
                    String line = " ";

                    while ((line = connection_reader.readLine()) != null) {
                        Log.d("print 2", line);
                        sb.append(line);
                        break;
                    }
                    connection_reader.close();
                    contents = sb.toString();
                    Log.d("Contents", contents);
                    if (!sb.toString().isEmpty()) {
                        String[] modelsListString = contents.split("split");
                        for (String s : modelsListString) {
                            String[] model_doc = s.split("comma");
                            models.add(model_doc);
                        }
                        String[] modelNames = new String[models.size()];
                        float[] modelWeights = new float[models.size()];
                        float[] modelThresholds = new float[models.size()];
                        byte[][] modelContent = new byte[models.size()][];
                        int version = Integer.parseInt(models.get(0)[1]);
                        for (int i = 0; i < models.size(); i++) {
                            modelNames[i] = models.get(i)[0].replace("'", "").replaceAll("/data/", "_") + "_" + models.get(i)[1] + ".tflite";
                            modelWeights[i] = Float.parseFloat(models.get(i)[3].replaceAll("'", ""));
                            String fileContents = models.get(i)[2].replaceAll("'", "");
                            modelThresholds[i] = Float.parseFloat(models.get(i)[4].replaceAll("'", ""));
                            Log.e(TAG, "******" + fileContents);
                            Log.e(TAG, "******" + modelWeights[i]);

                            String[] data = fileContents.split(",");
                            byte[] bytes = new byte[data.length];
                            for (int j = 0; j < data.length; j++) {
                                int a = Integer.parseInt(data[j]) & 0xff;
                                bytes[j] = (byte) a;
                            }
                            modelContent[i] = bytes;
                        }
                        /**
                         * Updating the ModelConfig object with newly downloaded models.
                         */
                        if (models.size() > 0 && models.get(0) != null && !models.get(0).equals("")) {
                            config.isDownloaded = true;
                            config.modelNames = modelNames;
                            config.modelWeights = modelWeights;
                            config.modelContent = modelContent;
                            config.oldDocID = oldDocID;
                            config.modelVersion = version;
                            config.thresholds = modelThresholds;
                            Log.e(TAG, "Downloaded new model successfully");
                        }

                    }
                }
            }



        } catch (Exception e) {
            config.newDocID = oldDocID;
            Log.e(TAG, "Error" + e);
        }
        return config;

    }
}