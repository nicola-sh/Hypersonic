<?php
define('IPOL_ALI_MODULE', 'ipol.aliexpress');
define('IPOL_ALI_CACHE_TIME', 86400);

/**
 * @deprecated 
 */
define('IPOLH_ALI_MODULE', IPOL_ALI_MODULE);

/**
 * @deprecated
 */
define('IPOLH_ALI_CACHE_TIME', IPOL_ALI_CACHE_TIME);

\Bitrix\Main\Loader::includeModule('sale');
\Bitrix\Main\Loader::includeModule('iblock');
\Bitrix\Main\Loader::includeModule('catalog');

define('TOP_AUTOLOADER_PATH', __DIR__ .'/vendor/taobao/');

require_once __DIR__ .'/vendor/taobao/Autoloader.php';

\CJSCore::RegisterExt('ipol_ali_admin_order_edit', array(
	'js'   => '/bitrix/js/ipol.aliexpress/order-admin-detail.js',
	'lang' => '/bitrix/modules/ipol.aliexpress/lang/'. LANGUAGE_ID .'/js/order-admin-detail.php',
	'rel'  => array('ajax' ,'popup')
));