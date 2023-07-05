<?php
return array(
	array(
		'callback' => '\\Ipol\\AliExpress\\Agent\\Order::syncOrderList();',
		'interval' => 1800,
	),

	array(
		'callback' => '\\Ipol\\AliExpress\\Agent\\Order::syncOrderStatus();',
		'interval' => 600,
	),
);