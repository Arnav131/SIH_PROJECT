package com.roadai.androidtest.sensors

import okhttp3.HttpUrl.Companion.toHttpUrl
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * Guards the receiver-URL rules. The app previously handed OkHttp a ws:// URL, which HttpUrl
 * rejects, so the socket was never dialled and no data ever reached the laptop.
 */
class EndpointTest {

 @Test fun plainHttpBaseUrlGetsTheDataPath() {
  val e = deriveEndpoint("http://192.168.1.5:8000")
  assertEquals("http://192.168.1.5:8000/ws/data", e.socket)
  assertEquals("ws://192.168.1.5:8000/ws/data", e.display)
 }

 @Test fun trailingSlashIsIgnored() {
  assertEquals("http://192.168.1.5:8000/ws/data", deriveEndpoint("http://192.168.1.5:8000/").socket)
 }

 @Test fun httpsMapsToWss() {
  val e = deriveEndpoint("https://receiver.example.com")
  assertEquals("https://receiver.example.com/ws/data", e.socket)
  assertEquals("wss://receiver.example.com/ws/data", e.display)
 }

 @Test fun wsInputIsAcceptedAndDialledOverHttp() {
  val e = deriveEndpoint("ws://192.168.1.5:8000/ws/data")
  assertEquals("http://192.168.1.5:8000/ws/data", e.socket)
  assertEquals("ws://192.168.1.5:8000/ws/data", e.display)
 }

 @Test fun wssInputIsDialledOverHttps() {
  assertEquals("https://host/ws/data", deriveEndpoint("wss://host/ws/data").socket)
 }

 @Test fun customPathIsPreserved() {
  assertEquals("http://host:9000/collect", deriveEndpoint("http://host:9000/collect").socket)
 }

 /** The regression itself: every derived socket URL must survive HttpUrl parsing. */
 @Test fun everySocketUrlIsParseableByOkHttp() {
  val inputs = listOf(
   "http://192.168.1.5:8000",
   "http://192.168.1.5:8000/",
   "ws://192.168.1.5:8000",
   "ws://192.168.1.5:8000/ws/data",
   "wss://receiver.example.com",
   "https://receiver.example.com/ws/data"
  )
  for (raw in inputs) {
   val url = deriveEndpoint(raw).socket
    .toHttpUrl()
    .newBuilder()
    .addQueryParameter("token", "urbansenseai_demo_token")
    .build()
   assertTrue(raw, url.toString().contains("token=urbansenseai_demo_token"))
   assertTrue(raw, url.scheme == "http" || url.scheme == "https")
  }
 }

 @Test fun blankUrlIsRejected() {
  try {
   deriveEndpoint("   ")
   fail("blank URL should be rejected")
  } catch (e: IllegalArgumentException) {
   assertEquals("Receiver URL required", e.message)
  }
 }

 /** People copy "192.168.1.5:8000" off the server banner; assume http rather than failing. */
 @Test fun schemelessUrlDefaultsToHttp() {
  val e = deriveEndpoint("192.168.1.5:8000")
  assertEquals("http://192.168.1.5:8000/ws/data", e.socket)
  assertEquals("ws://192.168.1.5:8000/ws/data", e.display)
 }

 @Test fun schemelessHostWithoutPortStillWorks() {
  assertEquals("http://my-laptop/ws/data", deriveEndpoint("my-laptop").socket)
 }

 @Test fun garbageUrlGivesAnActionableMessage() {
  try {
   deriveEndpoint("http://a b c")
   fail("a malformed URL should be rejected")
  } catch (e: IllegalArgumentException) {
   assertTrue(e.message, e.message!!.contains("http://192.168.1.5:8000"))
  }
 }

 @Test fun unsupportedSchemeIsRejected() {
  try {
   deriveEndpoint("ftp://192.168.1.5:8000")
   fail("ftp should be rejected")
  } catch (e: IllegalArgumentException) {
   assertTrue(e.message, e.message!!.contains("http(s) or ws(s)"))
  }
 }
}
