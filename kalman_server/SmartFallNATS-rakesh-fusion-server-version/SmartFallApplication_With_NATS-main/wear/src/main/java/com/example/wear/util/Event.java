package com.example.wear.util;

import java.nio.ByteBuffer;
import java.sql.Timestamp;
import java.util.UUID;

//import edu.txstate.reu.ble.BleProfile;

/**
 *  Represents an Event proceeding the reception of Bluetooth Low Energy data. An event is used in
 *  order to best describe the data received via BLE.
 *
 *  @author Colin Campbell (c_c953)
 *  @version 1.0
 *  @since 2021.12.13
 */
public class Event {

    /**
     *  A string representing the path associated with the event
     */
    protected String path = "";

    /**
     *  An array of float data sent via ble
     */
    protected float[] data;

    /**
     *  A timestamp representing the event's arrival time
     */
    protected Timestamp timestamp;

    /**
     *  A constructor which uses it's received parameters to create an instance
     *  of the Event class.
     *
     *  @param data byte: The received ble data
     *  @param timestamp Timestamp: The arrival time of the received data
     *  @param uuid UUID: The origin of the received data
     */
    public Event (byte[] data, Timestamp timestamp, UUID uuid) {
        this.data = cleanData(data);
        this.timestamp = timestamp;
        this.path = uuid.toString();
    }

    /**
     *  A method for converting an array of bytes received via ble into the corresponding float
     *  representation.
     *
     *  @param data byte: The data to converted to float values
     *  @return float: The corresponding float array associated with the received byte data
     */
    private float[] cleanData(byte[] data) {
        ByteBuffer buffer = ByteBuffer.wrap(data);

        float[] result = new float[data.length / 4];
        for (int i=0; i < result.length; i++) {
            result[i] = buffer.getFloat();
        }

        return result;
    }

    /**
     *  A method for accessing the event's path
     *
     *  @return String: The corresponding String belonging to the event's path attribute
     */
    public String getPath() { return path; }

    /**
     *  A method for accessing the event's data array
     *
     *  @return float: The corresponding float array belonging to the event's data attribute
     */
    public float[] getData() { return data; }

    /**
     *  A method for accessing the event's timestamp
     *
     *  @return Timestamp: The corresponding Timestamp belonging to the event's timestamp attribute
     */
    public Timestamp getTimestamp() { return timestamp; }
}

