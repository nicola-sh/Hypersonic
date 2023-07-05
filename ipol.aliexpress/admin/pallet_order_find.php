<?php
use Bitrix\Main\Loader;

require_once($_SERVER["DOCUMENT_ROOT"]."/bitrix/modules/main/include/prolog_admin.php");

Loader::includeModule('ipol.aliexpress');

$table = new \Ipol\Aliexpress\Admin\Table\PalletOrderFind();
$table->setDefaultFilterValues([
    'PALLET_ID'      => 0,
    '!ALI_PARCEL_ID' => 0,
    // 'SERVICE_VARIANT' => $_REQUEST['serviceVariant'],
    // 'DROP_POINT'      => $_REQUEST['serviceVariant'] == 'DropOff' ? $_REQUEST['dropPoint'] : null,
]);

$APPLICATION->SetTitle($table->getTitle());

print $table->render([
    'SHOW_FILTER' => 'Y',
    'SHOW_BUTTONS' => 'N',
]);

require_once($_SERVER["DOCUMENT_ROOT"]."/bitrix/modules/main/include/epilog_admin.php");