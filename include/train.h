#pragma once
#define LOG(x) std::cout << x << std::endl
#include <iostream>
#include <cstring>

int train_init(const char*);
int train();
int cleanup();