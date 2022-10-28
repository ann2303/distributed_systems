#include <stdio.h>
#define SIZE 15

int items[SIZE], front = -1, rear = -1, is_free = 1;

void enQueue(int value) {
  if (rear == SIZE - 1)
    printf("Очередь заполнена\n");
  else {
    if (front == -1)
      front = 0;
    rear++;
    items[rear] = value;
    printf("Добавлено значение -> %d\n", value);
  }
}

int deQueue() {
  if (front == -1) {
    printf("Очередь пуста\n");
    return -1;
  } else {
    int res = items[front]; 
    printf("Удален элемент: %d\n", items[front]);
    front++;
    if (front > rear)
      front = rear = -1;
    return res;
  }
}


