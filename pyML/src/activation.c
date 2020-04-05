#include "../include/nn.h"

double Activate(int kind, double value) {

	double act_value;

	if (kind==LINEAR)  act_value = value;
	if (kind==RELU)    act_value = value>0.0 ? value : 0.0 ;
	if (kind==SIGMOID) act_value = 1.0 / (1.0 + exp(-value));
	if (kind==TANH)    act_value = tanh(value);

	return act_value;

}

double dActivate(int kind, double act_value, double d_act_value) {

	double d_value;

	if (kind==LINEAR)  d_value = d_act_value;
	if (kind==RELU)    d_value = act_value>0.0 ? d_act_value : 0.0 ;
	if (kind==SIGMOID) d_value = d_act_value * act_value * (1.0 - act_value);
	if (kind==TANH)    d_value = d_act_value * (1.0 - act_value * act_value);

	return d_value;

}
