from speechloom.streaming import EventQueue, UTF8Stream

queue = EventQueue(32)
decoder = UTF8Stream()
for byte in '你好，语音模型。'.encode('utf-8'):
    text = decoder.feed(bytes([byte]))
    if text:
        queue.push('text', text)
queue.push('text', decoder.finish())
queue.push('done')
for event in queue.drain():
    print(event)
