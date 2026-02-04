package edu.txstate.reu.smartfall_multi_model_native.Util;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.AsyncTask;
import android.util.Log;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Properties;

import javax.mail.Authenticator;
import javax.mail.Message;
import javax.mail.MessagingException;
import javax.mail.PasswordAuthentication;
import javax.mail.Session;
import javax.mail.Transport;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeMessage;

/*
 * Author Priyanka Srinivas (p_s231)
 * Date : August 2020
 * This class sends an email to the care taker i.e the Emergency contact mentioned during the profile creation
 * The function sendEmail is invoked when the user has fallen and needs help
 */


public class SendMessage{
    Session session = null;
    String  subject,textMessage;
    public static String sender="tsmrafee@gmail.com";
    public static String recipient="tsmrafee@gmail.com";
    public static String userName="test";
    SharedPreferences pref;
    SharedPreferences.Editor editor;

    /*
     * @param - application context , latitude and longitude of current location
     */
    public void sendEmail(String userName, String recipient, Double latitude , Double longitude) throws Exception {
        this.userName = userName;
        this.recipient = recipient;
        subject = "Urgent!! SmartFall App Notification";
        Log.d("NEED HELP", SendMessage.recipient);
        String Lat = latitude.toString() ;
        String Long =longitude.toString();
        textMessage = "I've fallen and I need help. My location is : http://www.google.com/maps/place/" + latitude.toString() +","+  longitude.toString();



        Properties props = new Properties();
        props.put("mail.smtp.host", "smtp.gmail.com");
        props.put("mail.smtp.socketFactory.port", "465");
        props.put("mail.smtp.socketFactory.class", "javax.net.ssl.SSLSocketFactory");
        props.put("mail.smtp.auth", "true");
        props.put("mail.smtp.port", "465");



        // 1. Login to Gmail.
        // 2.Access the URL  https://www.google.com/settings/security/lesssecureapps
        // 3.Select "Turn on"

        session = Session.getDefaultInstance(props, new Authenticator() {
            protected PasswordAuthentication getPasswordAuthentication() {
                return new PasswordAuthentication("tsmrafee@gmail.com", "kcdxgrqnpswihqft");
            }
        });
        SendMessage.RetreiveFeedTask task = new RetreiveFeedTask();
        task.execute();
    }


    class RetreiveFeedTask extends AsyncTask<String, Void, String> {

        @Override
        protected String doInBackground(String... params) {
            Date date = new Date();
            SimpleDateFormat formatter = new SimpleDateFormat("dd-MM-yyyy HH:mm:ss");

            try {
                Message message = new MimeMessage(session);
                message.setFrom(new InternetAddress(SendMessage.sender));
                message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(SendMessage.recipient));
                message.setSubject(subject);
                message.setContent(textMessage+ "<br>"+ "UserName: " + userName+ "<br>" +"Time : " + formatter.format(date)  , "text/html; charset=utf-8");

                Transport.send(message);
            } catch (MessagingException e) {
                Log.d("Exception", "Email sending exception occured.");
            } catch (Exception e) {
                Log.d("Exception", "Email sending exception occured.");
            }
            return null;
        }

        @Override
        protected void onPostExecute(String result) {
            Log.d("Status", "Message sent");
        }
    }
}
