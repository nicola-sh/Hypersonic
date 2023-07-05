<?php
use Bitrix\Main\Loader;

require_once($_SERVER["DOCUMENT_ROOT"]."/bitrix/modules/main/include/prolog_admin.php");

Loader::includeModule('ipol.aliexpress');

$table = new Ipol\Aliexpress\Admin\Table\Pallet();

echo $table->render(['SHOW_TITLE' => 'N']);

$GLOBALS['APPLICATION']->SetTitle( $table->getTitle() );
