package com.example.wear.Database;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.couchbase.lite.CouchbaseLite;
import com.couchbase.lite.CouchbaseLiteException;
import com.couchbase.lite.DataSource;
import com.couchbase.lite.DatabaseConfiguration;
import com.couchbase.lite.Database;
import com.couchbase.lite.Document;
import com.couchbase.lite.Meta;
import com.couchbase.lite.MutableArray;
import com.couchbase.lite.MutableDocument;
import com.couchbase.lite.Query;
import com.couchbase.lite.QueryBuilder;
import com.couchbase.lite.Result;
import com.couchbase.lite.ResultSet;
import com.couchbase.lite.SelectResult;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import com.example.wear.config.SmartFallConfig;

/**
 * This class provides a Couchbase Lite 2.8.0 utility API that allows an android mobile device to
 * create and manage Couchbase Lite documents on a localized Couchbase database/bucket.
 * Additionally, this class provides a method for inserting locally saved documents into
 * a bucket hosted on a remote Couchbase Server.
 *
 * @author Colin Campbell (c_c953)
 * @version 1.0
 * @since 2021.12.13
 */
public class Couchbase {

    /**
     * A string TAG for Debugging
     **/
    private static final String TAG = "Couchbase";


//    public static final String API_KEY = "5mArt_fA11";

    /**
     * A constant string representing the name of the local Couchbase Lite bucket on the device
     */
    private static final String LOCAL_BUCKET_NAME = "Collection";

    /**
     * A URL object used for uploading documents to a remote Couchbase Server.
     */
    private static URL url;
    private static Context context;

    /**
     * A method for initializing the Couchbase Lite SDK and instantiating the URL object using
     * the UPLOAD_SERVER_URL string constant.
     *
     * @param mcontext Context: App context
     */
    public static void initialize (Context mcontext) throws IOException {
        CouchbaseLite.init(mcontext);
        context = mcontext;
        url = new URL(SmartFallConfig.getCloudConfig().dataUpload);
    }

    /**
     * A method that takes a JSON document, converts it into a Couchbase MutableDocument and saves
     * it into the local Couchbase Lite bucket.
     *
     * @param document JSONObject: The document to be saved to the local bucket
     * @return String: A Couchbase Lite document id associated with the inserted document
     */
    public static String insertDocument(JSONObject document) throws CouchbaseLiteException, IOException {
        DatabaseConfiguration databaseConfiguration = new DatabaseConfiguration();
        Database database = new Database(LOCAL_BUCKET_NAME, databaseConfiguration);
        Map<String,Object> prop = new ObjectMapper().readValue(document.toString(), HashMap.class);
        MutableDocument mutableDocument = new MutableDocument(prop);
        database.save(mutableDocument);

        return mutableDocument.getId();
    }

    public static String insertDocument(MutableDocument document) throws CouchbaseLiteException, IOException {
        DatabaseConfiguration databaseConfiguration = new DatabaseConfiguration();
        Database database = new Database(LOCAL_BUCKET_NAME, databaseConfiguration);

        database.save(document);
        Log.v(TAG,document.getId());


        return document.getId();
    }

    public static Document fetchDocument(String docId) throws CouchbaseLiteException, IOException {
        DatabaseConfiguration databaseConfiguration = new DatabaseConfiguration();
        Database database = new Database(LOCAL_BUCKET_NAME, databaseConfiguration);

        return database.getDocument(docId);
    }

    /**
     * A method for modifying the contents of a pre-existing document saved within the local
     * Couchbase Lite bucket. This method uses the Couchbase Lite document ID to retrieve the
     * document being updated and applies a JSON object of updates before saving the document back
     * to the database.
     *
     * @param documentId String: A value representing the Couchbase Lite document ID belonging to
     * the document needing to be modified
     * @param updates JSONObject: An object comprised of the modifications to be made to the
     * pre-existing document.
     */
    public static void updateDocument(String documentId, JSONObject updates) throws CouchbaseLiteException, JSONException {
        if (documentId == null || documentId.trim().isEmpty()) {
            android.util.Log.w("Couchbase", "updateDocument: documentId is null/empty; aborting");
            return;
        }
        DatabaseConfiguration databaseConfiguration = new DatabaseConfiguration();
        Database database = new Database(LOCAL_BUCKET_NAME, databaseConfiguration);


        Document document = database.getDocument(documentId);
        if (null != document) {
            MutableDocument mutableDocument = document.toMutable();

            Iterator<String> keys = updates.keys();

            while (keys.hasNext()) {
                String key = keys.next();
                Object value = updates.get(key);
                if (value instanceof JSONArray) {
                    MutableArray mutableArray = new MutableArray();
                    JSONArray jsonArray = updates.getJSONArray(key);
                    for (int i = 0; i < jsonArray.length(); i++) {
                        Object element = jsonArray.get(i);
                        mutableArray.insertValue(i, element);
                    }
                    mutableDocument.setArray(key, mutableArray);
                } else
                    mutableDocument.setValue(key, value);
            }
            database.save(mutableDocument);
        }
    }

    /**
     * A method for uploading the entire local bucket collection to a remote Couchbase Server.
     * This method sends a POST request to the url created from the UPLOAD_SERVER_URL constant.
     * The body of the POST request is a JSON object consisting on a bucket attribute and a
     * documents attribute. Following a successful response code from making the POST request,
     * all documents sent to the server are removed from the local bucket.
     */
    public static void uploadDocumentsToRemoteURL() throws IOException, JSONException, CouchbaseLiteException {

        ArrayList<JSONObject> documents = getAllDocuments();

        Log.d("Documents size: ", documents.size()+"");
        if (documents.size() != 0) {
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            ArrayList<JSONObject> body = new ArrayList<>();
            JSONObject jsonObject = new JSONObject();
            jsonObject.put("bucket", SmartFallConfig.REMOTE_BUCKET_NAME);
            jsonObject.put("documents", documents);
            body.add(jsonObject);
            connection.setDoOutput(true);
            connection.addRequestProperty("Authorization", SmartFallConfig.API_KEY);
            connection.getOutputStream().write(body.toString().getBytes());

            int responseCode = connection.getResponseCode();
            // if upload is successful purge all documents from the local database
            Log.d(TAG, "Data upload status : " + responseCode);
            if (responseCode == HttpURLConnection.HTTP_OK) {
                DatabaseConfiguration databaseConfiguration = new DatabaseConfiguration();
                Database database = new Database(LOCAL_BUCKET_NAME, databaseConfiguration);

                for (JSONObject document : documents) {
                    database.purge(document.getString("id"));
                }
            } else
                Log.e(TAG, "uploadDocumentsToRemoteURL response code: " + responseCode);
        }
    }

    public static void deletePreviousData() throws IOException, JSONException, CouchbaseLiteException{
        ArrayList<JSONObject> documents = getAllDocuments();
        DatabaseConfiguration databaseConfiguration = new DatabaseConfiguration();
        Database database = new Database(LOCAL_BUCKET_NAME, databaseConfiguration);

        for (JSONObject document : documents) {
            database.purge(document.getString("id"));
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.R)
    public static void writeDataInStorage() throws JSONException, CouchbaseLiteException {
        File folder = new File(context.getFilesDir()
                + "/Folder");

        boolean var = false;
        if (!folder.exists())
            var = folder.mkdir();

        Date now = new Date();
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMddHHmmss");
        String time = dateFormat.format(now);

        final String filename = folder.toString() + "/" + "Test"+ time +".csv";
        //System.out.println(filename);
        ArrayList<JSONObject> documents = getAllDocuments();
        System.out.println(documents.size());

        if (documents.size() != 0) {
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

                fw.append("outcome");
                fw.append(',');

                fw.append("negoutcome");
                fw.append(',');

                fw.append("subject-id");
                fw.append(',');

                fw.append('\n');

                DatabaseConfiguration databaseConfiguration = new DatabaseConfiguration();
                Database database = new Database(LOCAL_BUCKET_NAME, databaseConfiguration);

                for (JSONObject document : documents) {
                    JSONObject details = document.getJSONObject("keys");
                    if (details.getString("type").equals("tracker")) {
                        database.purge(document.getString("id"));
                        continue;
                    } else {
                        try {
                            System.out.println(details.toString());
                            String[] timestamps = details.getString("timestamp").split(",");
                            String[] acc_xs = details.getString("watch_accelerometer_x").split(",");
                            String[] acc_ys = details.getString("watch_accelerometer_y").split(",");
                            String[] acc_zs = details.getString("watch_accelerometer_z").split(",");

                            for (int i = 0; i < timestamps.length; i++) {
                                float timestamp = Float.parseFloat(timestamps[i]);
                                fw.append(timestamps[i]);
                                fw.append(',');

                                fw.append(acc_xs[i]);
                                fw.append(',');

                                fw.append(acc_ys[i]);
                                fw.append(',');

                                fw.append(acc_zs[i]);
                                fw.append(',');

                                if (details.getString("type").equals("TP")) {
                                    fw.append(1 + "");
                                    fw.append(',');

                                    fw.append(0 + "");
                                    fw.append(',');
                                } else {
                                    fw.append(0 + "");
                                    fw.append(',');

                                    fw.append(1 + "");
                                    fw.append(',');
                                }
                                fw.append(details.getString("uuid"));
                                fw.append(',');

                                fw.append('\n');
                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                        } finally {
//                            database.purge(document.getString("id"));
                        }
                    }
                }

                // fw.flush();
                fw.close();

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * A method for performing a query that returns all documents stored in the local Couchbase Lite
     * bucket.
     *
     * @return ArrayList of JSONObject: A list of documents where the 'id' attribute refers to a
     * document's document ID and the 'keys' attribute refers to the self document itself.
     */
    public static ArrayList<JSONObject> getAllDocuments() throws CouchbaseLiteException, JSONException {
        String id;
        JSONObject jsonObject, keys;
        ArrayList<JSONObject> documents = new ArrayList<>();

        DatabaseConfiguration databaseConfiguration = new DatabaseConfiguration();
        Database database = new Database(LOCAL_BUCKET_NAME, databaseConfiguration);
        Query query = QueryBuilder.select(SelectResult.expression(Meta.id), SelectResult.all()).from(DataSource.database(database));
        ResultSet results = query.execute();
        // int i=0;
        for (Result result : results.allResults()) {
//                System.out.println(result.toString());
            id =  result.getValue("id").toString();
            jsonObject = new JSONObject();
            keys = new JSONObject(result.getDictionary(LOCAL_BUCKET_NAME).toMap());
            jsonObject.put("id", id);
            jsonObject.put("keys", keys);

            if(!keys.getString("type").equals("???") && !keys.getString("type").equals("trackingCurrent")) {
                documents.add(jsonObject);
                // i++;
                Log.d("Document type", keys.getString("type"));
            }

            // if(i>3)
            //     break;

        }

        return documents;
    }



}
