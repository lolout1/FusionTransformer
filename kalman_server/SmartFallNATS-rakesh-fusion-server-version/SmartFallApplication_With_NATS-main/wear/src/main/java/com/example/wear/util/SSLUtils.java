package com.example.wear.util;

import javax.net.ssl.*;
import java.io.*;
import java.security.KeyStore;
import java.security.cert.Certificate;
import java.security.cert.CertificateFactory;

/* WORKING
public class SSLUtils {
    public static SSLContext createSSLContext(InputStream certInputStream) throws Exception {
        // Load the certificate
        CertificateFactory cf = CertificateFactory.getInstance("X.509");
        Certificate cert = cf.generateCertificate(certInputStream);

        // Create a KeyStore containing the certificate
        KeyStore keyStore = KeyStore.getInstance(KeyStore.getDefaultType());
        keyStore.load(null, null); // Initialize an empty KeyStore
        keyStore.setCertificateEntry("nats-cert", cert);

        // Create a TrustManager that trusts the certificate in the KeyStore
        TrustManagerFactory tmf = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
        tmf.init(keyStore);

        // Create the SSL context with the trust manager
        SSLContext sslContext = SSLContext.getInstance("TLS");
        sslContext.init(null, tmf.getTrustManagers(), null);

        return sslContext;
    }
}

 */

import javax.net.ssl.*;
import java.security.SecureRandom;
import java.security.cert.X509Certificate;

public class SSLUtils {
    public static SSLContext createInsecureSSLContext() throws Exception {
        // Create a TrustManager that does not validate certificate chains
        TrustManager[] trustAllCerts = new TrustManager[]{
                new X509TrustManager() {
                    @Override
                    public void checkClientTrusted(X509Certificate[] chain, String authType) {
                        // No validation
                    }

                    @Override
                    public void checkServerTrusted(X509Certificate[] chain, String authType) {
                        // No validation
                    }

                    @Override
                    public X509Certificate[] getAcceptedIssuers() {
                        return new X509Certificate[0];
                    }
                }
        };

        // Initialize the SSLContext with the custom TrustManager
        SSLContext sslContext = SSLContext.getInstance("TLS");
        sslContext.init(null, trustAllCerts, new SecureRandom());

        return sslContext;
    }
}
