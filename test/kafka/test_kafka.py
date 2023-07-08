from datetime import datetime
import numpy as np
import unittest
from confluent_kafka import Producer, Consumer
import time
import os

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@unittest.skipIf(IN_GITHUB_ACTIONS, 'Do not run on github actions.')
class TestKafka(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestKafka, self).__init__(*args, **kwargs)
        self.producer_config = {
            "bootstrap.servers": "localhost:9092",
            "enable.idempotence": True,
            "acks": "all",
            "retries": 100,
            "max.in.flight.requests.per.connection": 5,
            "compression.type": "snappy",
            "linger.ms": 5,
            "batch.num.messages": 32
        }
        self.producer = Producer(self.producer_config)

        self.producer_config = {
            "bootstrap.servers": "localhost:9092",
            "enable.idempotence": True,
            "acks": "all",
            "retries": 100,
            "max.in.flight.requests.per.connection": 5,
            "compression.type": "snappy",
            "linger.ms": 5,
            "batch.num.messages": 32
        }
        self.consumer = Consumer({
            'bootstrap.servers': "localhost:9092",
            'group.id': "my-consumer-group",
            'auto.offset.reset': "earliest"
        })
        self.topic="TestTopic"
    
    def test_producer(self):
        value = "Hello"
        print(f"Preparing to send '{value}' to topic {self.topic}")
        self.producer.produce(
            topic=self.topic, 
            value=(value), 
            timestamp=int(time.time())
        )
        self.producer.flush()
        print(f"Waiting 2 seconds")
        time.sleep(2)
        self.consumer.subscribe([self.topic])
        counter = 0
        while True:
            counter+=1
            if counter == 10:
                assert False
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            print(f"Received the message {msg.value().decode('utf-8')} at {str(datetime.fromtimestamp(msg.timestamp()[1]))}")
            assert value == msg.value().decode('utf-8')
            self.consumer.close()
            break


if __name__ == '__main__':
  np.random.seed(1337)
  unittest.main(verbosity=2)