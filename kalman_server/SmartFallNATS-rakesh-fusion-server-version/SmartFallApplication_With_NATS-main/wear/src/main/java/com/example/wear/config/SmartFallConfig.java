package com.example.wear.config;

/**
 * This class is the single location to change personalization strategy or remote server configuration.
 * Any new type of strategy or new remote server configuration has to be added in this class
 *
 * @author Bhargav Balusu (b_b515)
 * @version 1.0
 * @since 2022.05.15
 */
public class SmartFallConfig {

    public static final String PREDICTION_API = "http://172.20.10.14:5000";

    /**
     * Below is the key where top level strategy can be switched.
     * To change strategy, just uncomment required line and comment current strategy.
     */
    public static final String MODEL_TYPE  = "PERSONALIZED";
    //public static final String MODEL_TYPE  = "LSTM";
    //public static final String MODEL_TYPE  = "DEFAULT";


    /**
     * Below is the key where personalization strategy can be switched.
     * To change strategy, just uncomment required line and comment current strategy.
     */
    //public static final String PERSONALIZED_STRATEGY  = "ENSEMBLE";
    public static final String PERSONALIZED_STRATEGY  = "LSTM";

    /**
     * Below is the key we used to specify from where we will use the model.
     * To change strategy, just uncomment required line and comment current strategy.
     */
//    public static final String MODEL_FROM  = "ONLINE";
    public static final String MODEL_FROM  = "OFFLINE";
    public static final String OFFLINE_MODEL_FILE  = "saved_model128.tflite"; // file name in assets folder
    public static final float OFFLINE_MODEL_THRESHOLD  = 0.9f;
    /**
     * This is the key that holds the information about which remote server has to be accessed.
     */
    public static String ENV_FLAG = "SMART_FALL";
//    public static String ENV_FLAG = "GANYMEDE";
    public static final String API_KEY = "s_f-Mb4x;ntAA&'5q6>)XE6s"; //for smart fall server
//    public static final String API_KEY = "5mArt_fA11"; //for ganymede

    /**
     * A constant string representing the name of the remote Couchbase Server bucket
     */
    public static final String REMOTE_BUCKET_NAME = "smart-fall-data"; //for smartfall server
//    private static final String REMOTE_BUCKET_NAME = "test"; //for ganymede server

    /**
     * A constant integer representing the maximum length of the alpha queue
     */
    public static final int ALPHA_LIMIT = 10; // Previously 20
    // public static final int ALPHA_LIMIT = 80;

    /**
     * A constant integer representing the maximum length of a beta queue.
     * It is also considered as the window size.
     */
    public static final int BETA_LIMIT = 128;

    public static final int STEP_SIZE = 5; // Previously 10
    // public static final int STEP_SIZE = 16; // 80 - 64

    /**
     * This is the key to check if it is first time install or not. This key is set to true in ProfileFragment.java
     * if the user fills profile section. This property is used in ModelDownloader.java
     */
    public static boolean isFirst = false;


    public static class CloudIPConfig {
        public String downloadModel=null;
        public String dataUpload = null;
        public String checkVersion = null;
    }

    /**
     * This function returns the configuration of the above mentioned remote server.
     * @return CloudIPConfig object.
     */

    public static CloudIPConfig getCloudConfig() {
        CloudIPConfig urlPaths = new CloudIPConfig();
        if (ENV_FLAG == "MERCURY") {
            System.out.println("It is Mercury URL Path");
            urlPaths.checkVersion = "http://eil.cs.txstate.edu/couchbase-prod/checkversion.php";
            urlPaths.downloadModel = "http://eil.cs.txstate.edu/couchbase-prod/downloadmodels.php";
            urlPaths.dataUpload = "http://eil.cs.txstate.edu/couchbase-prod/uploadcouch.php";
        }
        else if (ENV_FLAG == "GANYMEDE") {
            System.out.println("It is Ganymede URL Path");
            urlPaths.checkVersion = "http://ganymede.cs.txstate.edu/checkversion.php";
            urlPaths.downloadModel = "http://ganymede.cs.txstate.edu/downloadmodels.php";
            urlPaths.dataUpload = "http://ganymede.cs.txstate.edu/upsertdocuments.php";
        }
        else if (ENV_FLAG == "EUROPA") {
            System.out.println("It is Europa URL Path");
            urlPaths.checkVersion = " http://eil.cs.txstate.edu/couchbase-dev2/checkversion.php";
            urlPaths.downloadModel = " http://eil.cs.txstate.edu/couchbase-dev2/downloadmodels.php";
            urlPaths.dataUpload = " http://eil.cs.txstate.edu/couchbase-dev2/uploadcouch.php";
        }else if (ENV_FLAG == "SMART_FALL") {
            System.out.println("It is smartfall server URL Path");
            urlPaths.checkVersion = "https://cssmartfall1.cose.txstate.edu/smartFall/checkversion.php";
            urlPaths.downloadModel = "https://cssmartfall1.cose.txstate.edu/smartFall/downloadmodels.php";
            urlPaths.dataUpload = "https://cssmartfall1.cose.txstate.edu/smartFall/upsertdocuments.php";
        }

        return urlPaths;
    }

}
