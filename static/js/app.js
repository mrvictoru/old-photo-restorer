@@ -160,7 +160,14 @@ function updateStatusUI(item) {
-  statusText.textContent = `Status: ${item.status}`;
+  // show translated status text
+  const statusKey = item.status === 'idle'
+    ? 'status_idle'
+    : item.status === 'running'
+      ? 'status_running'
+      : item.status === 'done'
+        ? 'status_done'
+        : '';
+  if (statusKey) statusText.textContent = translations[statusKey] || statusText.textContent;
