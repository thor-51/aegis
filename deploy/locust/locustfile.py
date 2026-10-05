from locust import HttpUser, task, between

class MicroserviceUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def call_api(self):
        self.client.get("/")
