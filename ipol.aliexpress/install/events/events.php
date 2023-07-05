<?php
return array(
	array(
		'module'   => 'sale',
		'name'     => 'OnSaleOrderBeforeSaved',
		'callback' => array('\\Ipol\\AliExpress\\EventListener', 'OrderBeforeSavedHandler'),
		'sort'     => 100,
		'path'     => '',
		'args'     => array(),
	),
	
	array(
		'module'   => 'sale',
		'name'     => 'OnSaleOrderSaved',
		'callback' => array('\\Ipol\\AliExpress\\EventListener', 'OrderAddHandlerOrder'),
		'sort'     => 100,
		'path'     => '',
		'args'     => array(),
	),

    array(
        'module'    => 'main',
        'name'      => 'onEpilog',
        'callback'  => array('\\Ipol\\AliExpress\\EventListener', 'OnAdminEpilog'),
        'sort'      => 100,
        'path'      => '',
        'args'      => array()
	),
	
	array(
		'module'   => 'main',
		'name'     => 'OnBuildGlobalMenu',
		'callback' => array('\\Ipol\\AliExpress\\EventListener', 'OnBuildGlobalMenu'),
		'sort'     => 100,
		'path'     => '',
		'args'     => array(),
	),
);