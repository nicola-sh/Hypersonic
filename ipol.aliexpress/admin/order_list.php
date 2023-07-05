<?php
use Bitrix\Main\Loader;

require_once($_SERVER["DOCUMENT_ROOT"]."/bitrix/modules/main/include/prolog_admin.php");

Loader::includeModule('ipol.aliexpress');

$table = new Ipol\Aliexpress\Admin\Table\Order();

echo $table->render([
    'SHOW_TITLE' => 'N',
    'TABLE' => [
        'SHOW_ROW_CHECKBOXES'       => true,
        'SHOW_CHECK_ALL_CHECKBOXES' => true,
    ]
]);

$GLOBALS['APPLICATION']->SetTitle( $table->getTitle() );
