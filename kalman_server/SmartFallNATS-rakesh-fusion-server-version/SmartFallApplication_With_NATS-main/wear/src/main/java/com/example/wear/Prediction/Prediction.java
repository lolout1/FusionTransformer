package com.example.wear.Prediction;

import android.content.Context;
import android.util.Log;

import com.couchbase.lite.Document;
import com.couchbase.lite.MutableDocument;

import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.TimeZone;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;
import java.util.Date;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ThreadLocalRandom;

//import edu.txstate.reu.ble.Event;
import com.example.wear.Database.Couchbase;
import com.example.wear.Database.Database;
import com.example.wear.config.ModelConfig;
import com.example.wear.config.SmartFallConfig;
import com.example.wear.util.Event;


/**
 *
 *
 * **/
public class Prediction {

    /**
     * A string TAG for Debugging
     */
    private static final String TAG = "Prediction";

    public static int num_pred = 0;

    /**
     * A queue of beta queues
     */
    private static ConcurrentLinkedQueue<ConcurrentLinkedQueue<Event>> alphaQueue;

    /**
     * A queue of received events awaiting prediction
     */
    private static ConcurrentLinkedQueue<Event> betaQueue;

    /**
     * A queue of heuristics parallel to the alpha queue
     */
    private static ConcurrentLinkedQueue<Float> heuristicsQueue;

    /**
     * A constant integer representing the maximum length of the alpha queue
     */
    private static int ALPHA_LIMIT;

    /**
     * A constant integer representing the maximum length of a beta queue
     */
    private static int BETA_LIMIT;

    private static int STEP_SIZE;

    /**
     * A constant float representing the threshold used to make a binary prediction
     */
    private static float THRESHOLD = 0.9f;

    /**
     * A boolean flag indicating when the threshold has been exceeded
     */
    private static boolean threshHoldExceeded = false;

    /**
     * A list of float values associated with various true negative predictions
     */
    private static ArrayList<float[]> trueNegativeData;

    /**
     * A list of Timestamp values associated with various true negative predictions
     */
    private static ArrayList<Timestamp> trueNegativeTimestamps;

    /**
     * A list of float values associated with a prediction exceeding the threshold value
     */
    private static ArrayList<float[]> exceededData;

    /**
     * A list of Timestamp values associated with a prediction exceeding the threshold value
     */
    private static ArrayList<Timestamp> exceededTimestamps;

    private static JSONObject tracker;



    private static Context cContext;

    /**
     * This method initializes the TensorFlowLite runtime and instantiates the class data structures
     *
     * @param context Context: App context
     * @throws IOException
     */
    public static void initialize (Context context) throws IOException {
        cContext = context;
        PredictionContext.initializeModel(context);
        alphaQueue = new ConcurrentLinkedQueue<ConcurrentLinkedQueue<Event>>();
        betaQueue = new ConcurrentLinkedQueue<Event>();
        heuristicsQueue = new ConcurrentLinkedQueue<Float>();
        trueNegativeData = new ArrayList<>();
        trueNegativeTimestamps = new ArrayList<>();
        exceededData = new ArrayList<>();
        exceededTimestamps = new ArrayList<>();
        THRESHOLD = PredictionContext.getThreshold();
        ALPHA_LIMIT = SmartFallConfig.ALPHA_LIMIT;
        BETA_LIMIT = SmartFallConfig.BETA_LIMIT;
        STEP_SIZE = SmartFallConfig.STEP_SIZE;
        Log.d("Threshold", ""+THRESHOLD);
    }

    /**
     *
     * @param event
     * @throws Exception
     */
    public static void makePrediction(Event event) throws Exception {

        float output = 0.0f;

        // perform final prediction if the alpha queue surpasses the threshold
        if (alphaQueue.size() >= ALPHA_LIMIT){

            ConcurrentLinkedQueue<Float> tempOueue = new ConcurrentLinkedQueue<>(heuristicsQueue);
            while (!tempOueue.isEmpty()) {
                output += tempOueue.poll();
            }

            output = output / ALPHA_LIMIT;

            // Log.d(TAG, String.valueOf(output) + " " + THRESHOLD + " " + heuristicsQueue.size() + " " + alphaQueue.size() );
            // Log.d(TAG, THRESHOLD + " " + heuristicsQueue.size() + " " + alphaQueue.size() );

            //FOR TESTING
            //output=0;
            if (output < THRESHOLD) {
                threshHoldExceeded = false;

                // Database
                //System.out.println("trueNegativeData.size() : " + trueNegativeData.size());
                if (trueNegativeData.size() > 375) {
                    // Create document and save to database
                    System.out.println("updating document!!!");
                    MutableDocument doc = createTestDocument(trueNegativeData,trueNegativeTimestamps,"TN");
                    System.out.println("calling for TN");
                    Database.insertDocument(doc);

                    trueNegativeData.clear();
                    trueNegativeTimestamps.clear();
                }

                for(int i=0; i< STEP_SIZE; i++) {
                    trueNegativeData.add(alphaQueue.peek().peek().getData());
                    trueNegativeTimestamps.add(alphaQueue.poll().poll().getTimestamp());

                    heuristicsQueue.poll();
                }
                // trueNegativeData.add(alphaQueue.peek().peek().getData());
                // trueNegativeTimestamps.add(alphaQueue.poll().poll().getTimestamp());
                // heuristicsQueue.poll();

            }
            else if (output >= THRESHOLD && !threshHoldExceeded) {

                threshHoldExceeded = true;

                // signal the watch to pause the sensor service

                while (!alphaQueue.isEmpty()) {
                    // add the final beta queue to the arrays
                    if(alphaQueue.size() == 1) {
                        ConcurrentLinkedQueue<Event> tempBetaQueue = alphaQueue.poll();
                        while (!tempBetaQueue.isEmpty()) {
                            exceededData.add(tempBetaQueue.peek().getData());
                            exceededTimestamps.add(tempBetaQueue.poll().getTimestamp());
                        }
                    }
                    else {
                        // add the oldest beta queue data to the arrays
                        exceededData.add(alphaQueue.peek().peek().getData());
                        exceededTimestamps.add(alphaQueue.poll().poll().getTimestamp());
                    }
                }
                // Create document and send to database
                MutableDocument doc = createTestDocument(exceededData, exceededTimestamps, "???");
                Database.insertDocument(doc);
                exceededData.clear();
                exceededTimestamps.clear();
                reset();
            }
            else {
                threshHoldExceeded = false;

                while (!alphaQueue.isEmpty()) {
                    trueNegativeData.add(alphaQueue.peek().peek().getData());
                    trueNegativeTimestamps.add(alphaQueue.poll().poll().getTimestamp());
                }

                while (!heuristicsQueue.isEmpty())
                    heuristicsQueue.poll();
            }

            num_pred++;
        }

        if (betaQueue.size() >= BETA_LIMIT) {

            alphaQueue.offer(betaQueue);
            ConcurrentLinkedQueue<Event> tempQueue = new ConcurrentLinkedQueue<>(betaQueue);
            //saveData(betaQueue);
            float betaSamples [][] = new float[tempQueue.size()][];
            for (int i = 0; i < betaSamples.length; i++) {
                betaSamples[i] = tempQueue.poll().getData();
            }
            float[][] accInput = new float[betaSamples.length][3];
            float[][] gyroInput = new float[betaSamples.length][3];

            for (int i = 0; i < betaSamples.length; i++) {
                accInput[i][0] = betaSamples[i][0];
                accInput[i][1] = betaSamples[i][1];
                accInput[i][2] = betaSamples[i][2];
                gyroInput[i][0] = betaSamples[i][3];
                gyroInput[i][1] = betaSamples[i][4];
                gyroInput[i][2] = betaSamples[i][5];
            }

            heuristicsQueue.offer(PredictionContext.makeInference(accInput, gyroInput));

            betaQueue.poll(); // remove oldest from beta
        }
        betaQueue.offer(event);
    }

    public static void onFellClickedInWatch() throws Exception {
        Log.w(TAG, "Manual FALL button pressed on watch!");
        while (!alphaQueue.isEmpty()) {
            // add the final beta queue to the arrays
            if(alphaQueue.size() == 1) {
                ConcurrentLinkedQueue<Event> tempBetaQueue = alphaQueue.poll();
                while (!tempBetaQueue.isEmpty()) {
                    exceededData.add(tempBetaQueue.peek().getData());
                    exceededTimestamps.add(tempBetaQueue.poll().getTimestamp());
                }
            }
            else {
                // add the oldest beta queue data to the arrays
                exceededData.add(alphaQueue.peek().peek().getData());
                exceededTimestamps.add(alphaQueue.poll().poll().getTimestamp());
            }
        }
        // Create document and send to database
        MutableDocument doc = createTestDocument(exceededData, exceededTimestamps, "FN"); // previously TP
        Database.insertDocument(doc);
        exceededData.clear();
        exceededTimestamps.clear();
        reset();
    }

    private static void saveData(ConcurrentLinkedQueue<Event> betaQueue){
        ConcurrentLinkedQueue<Event> tempQueue = new ConcurrentLinkedQueue<>(betaQueue);
        File folder = new File(cContext.getFilesDir()
                + "/Folder");

        boolean var = false;
        if (!folder.exists())
            var = folder.mkdir();

        Date now = new Date();
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmssSSSS");
        String time = dateFormat.format(now);

        final String filename = folder.toString() + "/" + "Test"+ time +".csv";
        System.out.println(filename);
        try {
            FileWriter fw = new FileWriter(filename);

            fw.append("timestamp");
            fw.append(',');

            fw.append("Acc_x");
            fw.append(',');

            fw.append("Acc_y");
            fw.append(',');

            fw.append("Acc_z");
            fw.append(',');

            fw.append('\n');

            while(!tempQueue.isEmpty()){
                try {
                    Event event = tempQueue.poll();
                    float[] data = event.getData();
                    String timestamp = event.getTimestamp().toString();
                    fw.append(timestamp);
                    fw.append(',');

                    fw.append(data[0]+"");
                    fw.append(',');

                    fw.append(data[1]+"");
                    fw.append(',');

                    fw.append(data[2]+"");
                    fw.append(',');

                    fw.append('\n');
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            // fw.flush();
            fw.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    /**
     * This method resets the prediction algorithm by clearing each of the queue data structures
     */
    public static void reset() {
        threshHoldExceeded = false;
        alphaQueue.clear();
        betaQueue.clear();
        heuristicsQueue.clear();
    }

    public static MutableDocument createTestDocument( ArrayList<float[]> dataArray,  ArrayList<Timestamp> timeStamps, String label){

        MutableDocument mutableDocument = null;

        try {
            ModelConfig config = ModelConfig.getModelConfig();
            Date now = new Date();
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmssSSS");
            dateFormat.setTimeZone(TimeZone.getTimeZone("UTC"));
            String time = dateFormat.format(now);
            int randomNum = ThreadLocalRandom.current().nextInt(1000, 9999);
            String id = String.format("UUID %s TIME %s RAND %s", config.uuid, time, randomNum);

            mutableDocument = new MutableDocument(id);
            //Log.d(TAG, "Document ID created: " + id);

            mutableDocument.setString("uuid",config.uuid);
            //Log.d(TAG, "UUID set in document: " + config.uuid);

            // Buffers for accelerometer and gyroscope data
            StringBuilder accX = new StringBuilder();
            StringBuilder accY = new StringBuilder();
            StringBuilder accZ = new StringBuilder();
            StringBuilder gyroX = new StringBuilder();
            StringBuilder gyroY = new StringBuilder();
            StringBuilder gyroZ = new StringBuilder();

            StringBuffer timeSTampsInMilli = new StringBuffer();



            mutableDocument.setString("comparableTime",time);
            //Log.d(TAG, "Comparable time set in document: " + time);
            //Log.d(TAG, "First timestamp set");

            mutableDocument.setLong("firstTimestamp", System.currentTimeMillis());
            timeSTampsInMilli.append(timeStamps.get(0).getTime());

            for(int i=1; i<timeStamps.size(); i++){
                timeSTampsInMilli.append(",");
                timeSTampsInMilli.append(timeStamps.get(i).getTime());
            }
            
            // First entry
            accX.append(dataArray.get(0)[0]);
            accY.append(dataArray.get(0)[1]);
            accZ.append(dataArray.get(0)[2]);
            gyroX.append(dataArray.get(0)[3]);
            gyroY.append(dataArray.get(0)[4]);
            gyroZ.append(dataArray.get(0)[5]);

            // Remaining entries
            for(int i=1; i<dataArray.size(); i++){
                float[] data = dataArray.get(i);
                if(data.length == 6){
                    accX.append(",").append(data[0]);
                    accY.append(",").append(data[1]);
                    accZ.append(",").append(data[2]);
                    gyroX.append(",").append(data[3]);
                    gyroY.append(",").append(data[4]);
                    gyroZ.append(",").append(data[5]);
                }
            }
            mutableDocument.setString("timestamp", timeSTampsInMilli.toString());
            //Log.d(TAG, "Timestamps in millis set: " + timeSTampsInMilli.toString());

            mutableDocument.setString("type", label);
            //Log.d(TAG, "Type set: " + label);

            mutableDocument.setString("model_version", ""+ModelConfig.getModelConfig().modelVersion);
            //Log.d(TAG, "Model version set: " + config.modelVersion);

            // Store values in document
            mutableDocument.setString("watch_accelerometer_x", accX.toString());
            mutableDocument.setString("watch_accelerometer_y", accY.toString());
            mutableDocument.setString("watch_accelerometer_z", accZ.toString());
            mutableDocument.setString("watch_gyroscope_x", gyroX.toString());
            mutableDocument.setString("watch_gyroscope_y", gyroY.toString());
            mutableDocument.setString("watch_gyroscope_z", gyroZ.toString());

            //Log.d(TAG, "Accelerometer data set");

            //Log.d(TAG, "tracker"+tracker);

            if(tracker != null){
                tracker.put("lastDoc", id);
                tracker.put("docId", "model-version-available-on-server");
                if(tracker.getString("firstDoc").equals("null")){
                    tracker.put("firstDoc",id);
                }
                Couchbase.updateDocument(config.trackerId+"local",tracker);
                //Log.d(TAG, "Tracker updated");

            }

        } catch(Exception e){
            Log.e(TAG,"Exception while creating test document  "  + e.getMessage());
        }
        return mutableDocument;

    }

    public static void uploadTracker(Context context){
        ModelConfig config = ModelConfig.getModelConfig(context);
        try{
            Document doc = Couchbase.fetchDocument(config.trackerId + "local");
            if(doc != null){
                MutableDocument document =new MutableDocument(config.trackerId);

                document.setString("firstDoc", doc.getString("firstDoc"));
                document.setString("lastDoc", doc.getString("lastDoc"));
                document.setString("docId", doc.getString("docId"));
                document.setString("type", "tracker");
                document.setString("uuid", doc.getString("uuid"));
                document.setString("version", doc.getString("version"));
                Couchbase.insertDocument(document);
            }

        } catch(Exception e){
            Log.e(TAG, "Exception while uploading tracker document : " + e.getMessage());
        }



    }



    public static  void updateTracker(Context context){
        ModelConfig config = ModelConfig.getModelConfig(context);
        try {

            if(config.trackerId == null){
                Date now = new Date();
                SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmssSSS");
                String time = dateFormat.format(now);
                String id = String.format("UUID %s TIME %s VERSION %s", config.uuid, time, config.modelVersion);
                MutableDocument document = new MutableDocument(id + "local");
                document.setString("firstDoc",null);
                document.setString("lastDoc",null);
                // System.out.println("DOC ID ::: "+config.newDocID);
                document.setString("docId",config.newDocID);
                //System.out.println("updated doc id ::: "+doc.getString("docId"));
                document.setString("type", "trackingCurrent");
                document.setString("uuid", config.uuid);
                document.setString("version", String.valueOf(config.modelVersion));
                Couchbase.insertDocument(document);
                tracker = new JSONObject(document.toMap());
                config.trackerId = id;
                PersonalizedPredictionLSTM.updateConfigFile(context);
            }else {
                Document doc = Couchbase.fetchDocument(config.trackerId + "local");
                if(doc == null || doc.getString("docId") == null || !doc.getString("docId").equals(config.newDocID)){
                    uploadTracker(context);
                    Date now = new Date();
                    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmssSSS");
                    String time = dateFormat.format(now);
                    String id = String.format("UUID %s TIME %s VERSION %s", config.uuid, time, config.modelVersion);
                    MutableDocument document = new MutableDocument(id + "local");
                    document.setString("firstDoc",null);
                    document.setString("lastDoc",null);
                    // System.out.println("DOC ID  in else ::: "+config.newDocID);
                    document.setString("docId",config.newDocID);
                    document.setString("type", "trackingCurrent");
                    document.setString("uuid", config.uuid);
                    document.setString("version", String.valueOf(config.modelVersion));
                    Couchbase.insertDocument(document);
                    tracker = new JSONObject(document.toMap());
                    config.trackerId = id;
                    PersonalizedPredictionLSTM.updateConfigFile(context);

                }else {
                    tracker = new JSONObject(doc.toMap());
                }
            }
        } catch (Exception e){
            e.printStackTrace();
        }

    }

}
